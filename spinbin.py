import os
import json
import pickle
import sqlite3
from hashlib import md5
from multiprocessing import Pool
from datetime import datetime, timedelta
from collections.abc import Iterable

from numpy import arange


class Spinbin():
    """ job scheduler utility featuring:
            - spatial indexing, binning, and hashing for spatial processing
            - in-memory storage of arbitrary binary objects
            - parallel process management (see inherited: ParallelSpinbin)
    
        intended to be a lightweight utility for hashmap-like boolean 
        indexing on function calls, performant storage of in-memory binary 
        objects mapped to boundary regions, and parallel process scheduling to 
        run spatial/temporal processing in smaller batches

        for usage: see __main__ demo at bottom of script
        requires python3.8
    """

    # compute 8-bit integer hash for a given dictionary
    hash_dict = lambda self, kwargs, seed='': int(md5((str(seed) + json.dumps(kwargs, sort_keys=True, default=str)).encode('utf-8')).hexdigest(), base=16) >> 80

    def __init__(self, /, *, dx=2, dy=2, dz=5000, dt=timedelta(days=1), storagedir=os.getcwd(), inmemory=False, bins=True, **kwargs): 
        """
            args:
                dx:
                    delta longitude bin size (int)
                dy: 
                    delta latitude bin size (int)
                dz:
                    delta depth bin size (int)
                dt:
                    delta time bin size (timedelta)
                storagedir:
                    filepath location to store checksums (string)
               **kwargs:
                    boundary arguments
        """
        assert os.path.isdir(str(storagedir)), f'invalid dir {storagedir}'
        assert kwargs != {} or bins == False, 'no boundaries provided'

        self.storagedir, self.inmemory = storagedir, inmemory
        self.storage = os.path.join(storagedir, 'checksums.db') if not inmemory else ':memory:'
        self.kwargslist = list(self.bin_kwargs(dx, dy, dz, dt, **kwargs)) if bins else [kwargs]

    def __enter__(self):
        with sqlite3.connect(self.storage) as con:
            db = con.cursor()
            db.execute('CREATE TABLE IF NOT EXISTS hashmap(hash INT NOT NULL, bytes BLOB)')
            db.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_map on hashmap(hash)')
        # TODO: 
        # if self.inmemory: copy disk database into memory
        # https://stackoverflow.com/questions/4019081/how-to-copy-a-sqlite-table-from-a-disk-database-to-a-memory-database-in-python
        assert not self.inmemory, 'feature not yet implemented'  
        return self

    def __call__(self, /, *, callback,  **passkwargs):  # https://docs.python.org/3/whatsnew/3.8.html
        return list(self.__call_generator__(callback=callback, **passkwargs))

    def __call_generator__(self, /, *, callback, **passkwargs):
        seed=f'{callback.__module__}.{callback.__name__}:{json.dumps(passkwargs, default=str, sort_keys=True)}'
        for kwargs in self.kwargslist: 
            if not self.serialized(kwargs, seed): self.insert_hash(kwargs, seed, callback(**passkwargs, **kwargs))
            elif self.inmemory: self.update_hash(kwargs, seed, callback(**passkwargs, **kwargs))
            yield pickle.loads(self.serialized(kwargs, seed))

    def __exit__(self, exc_type, exc_value, tb):
        # TODO: 
        # if self.inmemory: copy in-memory database back to disk
        # https://stackoverflow.com/questions/4019081/how-to-copy-a-sqlite-table-from-a-disk-database-to-a-memory-database-in-python
        assert not self.inmemory, 'feature not yet implemented'  

    def insert_hash(self, kwargs={}, seed='', obj=None):
        with sqlite3.connect(self.storage) as con:
            db = con.cursor()
            db.execute('INSERT INTO hashmap VALUES (?,?)', (self.hash_dict(kwargs, seed), pickle.dumps(obj)))

    def update_hash(self, kwargs={}, seed='', obj=None):
        with sqlite3.connect(self.storage) as con:
            db = con.cursor()
            db.execute('UPDATE hashmap SET bytes = ? WHERE hash = ?', (pickle.dumps(obj), self.hash_dict(kwargs, seed)))

    def serialized(self, kwargs={}, seed=''):
        """ returns binary object or True if hash exists in database, else False """
        with sqlite3.connect(self.storage) as con:
            db = con.cursor()
            db.execute('SELECT * FROM hashmap WHERE hash == ?', (self.hash_dict(kwargs, seed),))
            res = db.fetchone()
        if res is None: return False
        if res[1] is None: return True
        if res[1] is not None: return res[1]

    def bin_kwargs(self, dx, dy, dz, dt, /, **kwargs):
        """ generate argument sets as area subsets of boundary kwargs

            kwargs are split into dx° * dy° * dz° * dt bins.
            coordinates are rounded to nearest outer-boundary degree integer

            args:
                deltas: (float or timedelta)
                    bin size of axes

                **kwargs (dict) 
                    boundary arguments, e.g.
                    >>> kwargs = {
                    ... 'south': 43.21, 'north': 46.54, 'west': -123.45, 'east': -110.01, 'top': 0, 'bottom': 5000, 
                    ... 'start': datetime(2000, 1, 1, 0, 0), 'end': datetime(2000, 1, 2, 0, 0)}

            yields: 
                dictionary containing args as a subset of input boundaries
        """

        spacebins = lambda a, b, delta: arange(min(a,b)-(min(a,b)%(delta*1)), max(a,b)-(max(a,b)%(delta*-1)), delta)

        for axmin, axmax, delta in zip(('west','south','top','start',), ('east','north','bottom','end'), (dx,dy,dz,dt)): 
            if axmin not in kwargs.keys(): kwargs[axmin] = 0
            if axmax not in kwargs.keys(): kwargs[axmax] = 0
            if min(kwargs[axmin],kwargs[axmax]) == max(kwargs[axmin],kwargs[axmax]): kwargs[axmax] += delta

        # spin to win!
        for x in spacebins(kwargs['west'], kwargs['east'], dx):
            for y in spacebins(kwargs['south'], kwargs['north'], dy): 
                for z in spacebins(kwargs['top'], kwargs['bottom'], dz):
                    for t in arange(kwargs['start'].date(), kwargs['end'], dt).astype(datetime):
                        yield dict(zip(('west', 'east', 'south', 'north', 'top', 'bottom', 'start', 'end',), (x, x+dx, y, y+dy, z, z+dz, t, t+dt,)))


class ParallelSpinbin(Spinbin):
    """ run spinbin jobs in a parallel processing pool """

    def __init__(self, /, *, pool=1, dx=2, dy=2, dz=5000, dt=timedelta(days=1), storagedir=os.getcwd(), inmemory=False, bins=True, **kwargs): 
        """
            args:
                dx:
                    delta longitude bin size (int)
                dy: 
                    delta latitude bin size (int)
                dz:
                    delta depth bin size (int)
                dt:
                    delta time bin size (timedelta)
                storagedir:
                    filepath location to store checksums (string)
               **kwargs:
                    boundary arguments
        """
        assert os.path.isdir(storagedir)
        assert kwargs != {} or bins == False, 'no boundaries provided'

        self.pool, self.storagedir, self.inmemory, = pool, storagedir, inmemory
        self.storage = os.path.join(storagedir, 'checksums.db') if not inmemory else ':memory:'
        self.kwargslist = list(self.bin_kwargs(dx, dy, dz, dt, **kwargs)) if bins else [kwargs]

    def __call__(self, /, *, callback, **passkwargs):  # https://docs.python.org/3/whatsnew/3.8.html
        with Pool(self.pool) as p: 
            return list(p.map(self.__call_generator__, zip((callback for _ in self.kwargslist), self.kwargslist, (passkwargs for _ in self.kwargslist))))

    def __call_generator__(self, args):
        callback, kwargs, passkwargs = args
        seed=f'{callback.__module__}.{callback.__name__}:{json.dumps(passkwargs, default=str, sort_keys=True)}'
        if not self.serialized(kwargs, seed): self.insert_hash(kwargs, seed, callback(**passkwargs, **kwargs))
        elif self.inmemory: self.update_hash(kwargs, seed, callback(**passkwargs, **kwargs))
        return pickle.loads(self.serialized(kwargs, seed))


if __name__ == '__main__':
    """ run the demo to split boundary arguments into smaller areas, 
        then log the results of each function call

        >   python3.8 spinbin.py 


        run demo again to quickly load cached results

        >   python3.8 spinbin.py 
    """

    import time

    def callback(**kwargs):
        """ demo: some arbitrary slow process that accepts space/time boundaries as args """
        print(f'hello world! i am alive!\n{json.dumps(kwargs, default=str, indent=1)}')
        time.sleep(0.5)
        return str(datetime.now().time())

    def parallelized_callback(**kwargs):
        """ another useless demo function """
        print(f'hello world! i am parallel! {kwargs = }')  # https://docs.python.org/3/whatsnew/3.8.html
        time.sleep(1)

    # define some boundaries using or subsetting these dict keys (only used when bins=True)
    kwargs = {
            'west':    -123.45,     'east':    -110.01,
            'south':    43.21,      'north':    46.54, 
            'bottom':   5000,       'top':      0, 
            'start':    datetime(2000, 1, 1, 0, 0), 
            'end':      datetime(2000, 1, 2, 0, 0)
        }

    # here kwargs will split into 21 function calls using default spatial bin sizes
    with Spinbin(bins=True, inmemory=False, **kwargs) as scheduler: 
        results = scheduler(callback=callback, testarg='some arg', anotherarg='changing this will invalidate results hash')

    print(results)

    # and again, but this time in parallel
    with ParallelSpinbin(pool=10, **kwargs) as scheduler: 
        scheduler(callback=parallelized_callback, newargument='yerp')

