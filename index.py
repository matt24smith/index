''' job scheduler and hashmap database utility '''

import os
import json
import pickle
import sqlite3
import logging
from hashlib import md5
from multiprocessing import Pool
from datetime import datetime, timedelta

from numpy import arange


class index():
    ''' database and process management utility

        An SQL database for caching arbitrary binary.  Intended to be used as a
        wrapper for arbitrary function calls, where the result of the function
        is stored as a binary BLOB in the db. A hash of the function name,
        path, and input arguments will be interpreted as a 16-bit integer to be
        used as primary key on a clustered index.

        Optionally can also store True/False instead of binary, segment
        a bounding-box coordinate region into smaller partitions, and
        run tasks in parallel

        Example:

        >>> import os
        >>> import time
        >>> import json
        >>> from datetime import datetime, timedelta

        >>> def callback(**kwargs):
        ...     print(f'hello from {os.getpid()}\\n{json.dumps(kwargs, default=str, indent=1)}')
        ...     time.sleep(0.5)
        ...     return str(datetime.now().time())

        >>> def parallelized_callback(**kwargs):
        ...     print(f'hello from {os.getpid()}\t{kwargs}')
        ...     time.sleep(1)
        ...     return None


        define some boundaries using or subsetting these dict keys
        (only used when bins=True)

        >>> kwargs = {
        ...         'west':    -123.45,     'east':    -110.01,
        ...         'south':    43.21,      'north':    46.54,
        ...         'bottom':   5000,       'top':      0,
        ...         'start':    datetime(2000, 1, 1, 0, 0),
        ...         'end':      datetime(2000, 1, 2, 0, 0)
        ...         }

        here kwargs will define 21 function calls using default
        2-degree coordinate binning

        >>> with index(bins=True, store=True, **kwargs) as scheduler:
        ...     results = scheduler(callback=callback,
        ...                         testarg='some arg',
        ...                         anotherarg='args will salt hashes')
        >>> print(results)

        and again, but this time in parallel

        >>> with parallelindex(pool=10, **kwargs) as scheduler:
        ...     scheduler(callback=parallelized_callback, newargument='test')
    '''

    # each data blob in the database assumed to be uniquely described by a function
    # and its input arguments
    # seed used for hashing is a string representation of the function module path,
    # name, and optionally additional arguments
    hash_seed = lambda self, callback, passkwargs={
    }: f'{callback.__module__}.{callback.__name__}:{json.dumps(passkwargs, default=str, sort_keys=True)}'

    # compute 64-bit integer hash for a given dictionary
    # sqlite maxint: +/- (2^63)-1
    hash_dict = lambda self, kwargs, seed='': (int(md5((str(seed) + json.dumps(
        kwargs, sort_keys=True, default=str)).encode('utf-8')).hexdigest(),
                                                   base=16) >> 64) - (2**63
                                                                      ) - 1

    def __init__(self,
                 *,
                 pool=1,
                 store=False,
                 inmemory=False,
                 bins=True,
                 dx=2,
                 dy=2,
                 dz=5000,
                 dt=timedelta(days=1),
                 storagedir=os.getcwd(),
                 filename='checksums.db',
                 **kwargs):
        """
            args:
                pool:
                    number of processes to run in parallel. when not using the
                    parallelindex subclass, this must be equal to 1.
                store:
                    determines whether to store the results of the callback.
                    when False, no binary will be serialized. only a hash of the
                    function and input arguments will be stored to determine
                    whether the callback was called
                inmemory:
                    when store=True and inmemory=True, callback results can be
                    stored to an in-memory database. this allows for faster
                    read and update times for processing that requires frequent
                    updates. will be copied to disk upon __exit__
                bins:
                    if True, boundary kwargs supplied to __init__ will be
                    split into multiple smaller area bins and passed to the
                    callback function. when
                dx:
                    delta longitude bin size (float)
                dy:
                    delta latitude bin size (float)
                dz:
                    delta depth bin size (float)
                dt:
                    delta time bin size (timedelta)
                storagedir:
                    directory to store spatial hashes and binary objects
                kwargs:
                    boundary arguments to split into bins, e.g.
                    (south, west, north, east, top, bottom, start, end)
        """
        assert kwargs != {} or bins == False, 'no boundaries provided'
        self.store, self.pool, self.storagedir, self.inmemory, = store, pool, storagedir, inmemory
        self.storage = os.path.join(storagedir,
                                    filename) if not inmemory else ':memory:'
        self.kwargslist = list(self.bin_kwargs(dx, dy, dz, dt, **
                                               kwargs)) if bins else [kwargs]

    def __enter__(self):
        assert self.kwargslist != [], 'empty kwargs!'
        assert os.path.isdir(str(self.storagedir)), f'invalid dir {storagedir}'
        with sqlite3.connect(self.storage) as con:
            db = con.cursor()
            db.execute(
                'CREATE TABLE IF NOT EXISTS hashmap(hash INTEGER PRIMARY KEY, bytes BLOB) WITHOUT ROWID;'
            )
            #db.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_map on hashmap(hash)')
            minval = (int(''.join(['0'
                                   for _ in range(32)]), base=16) >> 64) - (2**
                                                                            63)
            maxval = (int(''.join(['f'
                                   for _ in range(32)]), base=16) >> 64) - (2**
                                                                            63)
            db.execute('INSERT OR IGNORE INTO hashmap VALUES (?,?)',
                       (minval, pickle.dumps(None)))
            db.execute('INSERT OR IGNORE INTO hashmap VALUES (?,?)',
                       (maxval, pickle.dumps(None)))
        # TODO:
        # if self.inmemory: copy disk database into memory
        # https://stackoverflow.com/questions/4019081/how-to-copy-a-sqlite-table-from-a-disk-database-to-a-memory-database-in-python
        assert not self.inmemory, 'feature not yet implemented'
        return self

    def __call__(self, *, callback, **passkwargs):
        return list(self.__call_generator__(callback=callback, **passkwargs))

    def __call_generator__(self, *, callback, **passkwargs):
        #seed=f'{callback.__module__}.{callback.__name__}:{json.dumps(passkwargs, default=str, sort_keys=True)}'
        seed = self.hash_seed(callback, passkwargs)
        assert self.pool == 1, 'use parallelindex for processing pool'
        for kwargs in self.kwargslist:
            if not self.serialized(kwargs, seed):
                self.insert_hash(kwargs, seed,
                                 callback(**passkwargs, **kwargs))
            elif self.inmemory:
                self.update_hash(kwargs, seed,
                                 callback(**passkwargs, **kwargs))
            yield pickle.loads(self.serialized(kwargs, seed))

    def __exit__(self, exc_type, exc_value, tb):
        # TODO:
        # if self.inmemory: copy in-memory database back to disk
        # https://stackoverflow.com/questions/4019081/how-to-copy-a-sqlite-table-from-a-disk-database-to-a-memory-database-in-python
        assert not self.inmemory, 'feature not yet implemented'

    def insert_hash(self, kwargs={}, seed='', obj=None):
        logging.debug(
            f'INSERT HASH {self.hash_dict(kwargs, seed)}\nseed = {seed}\nBIN: kwargs = {kwargs}'
        )
        with sqlite3.connect(self.storage) as con:
            db = con.cursor()
            db.execute(
                'INSERT INTO hashmap VALUES (?,?)',
                (self.hash_dict(kwargs, seed),
                 bytes(
                     pickle.dumps(obj) if self.store else pickle.dumps(None))))

    def update_hash(self, kwargs={}, seed='', obj=None):
        logging.debug(
            f'UPDATE HASH {self.hash_dict(kwargs, seed)}\nseed = {seed }\nBIN: kwargs = {kwargs}'
        )
        with sqlite3.connect(self.storage) as con:
            db = con.cursor()
            db.execute('UPDATE hashmap SET bytes = ? WHERE hash = ?',
                       (pickle.dumps(obj), self.hash_dict(kwargs, seed)))

    def drop_hash(self, kwargs={}, seed=''):
        logging.debug(
            f'DROP HASH {self.hash_dict(kwargs, seed)}\nseed = {seed }\nBIN: kwargs = {kwargs}'
        )
        with sqlite3.connect(self.storage) as con:
            db = con.cursor()
            db.execute('DELETE FROM hashmap WHERE hash = ?',
                       (self.hash_dict(kwargs, seed), ))

    def serialized(self, kwargs={}, seed=''):
        """ returns binary object or True if hash exists in database, else False """
        with sqlite3.connect(self.storage) as con:
            db = con.cursor()
            db.execute('SELECT * FROM hashmap WHERE hash == ?',
                       (self.hash_dict(kwargs, seed), ))
            res = db.fetchone()
        logging.debug(
            f'CHECK HASH {self.hash_dict(kwargs, seed)}: {"exists!" if res is not None else "missing!" }\nseed = {seed}\nBIN: kwargs = {kwargs}'
        )
        if res is None: return False
        if res[1] is None: return True
        if res[1] is not None: return res[1]

    def bin_kwargs(self, dx, dy, dz, dt, **kwargs):
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
        if 'start' not in kwargs.keys(): kwargs['start'] = datetime(2000, 1, 1)
        if 'end' not in kwargs.keys(): kwargs['end'] = kwargs['start'] + dt

        for axmin, axmax, delta in zip(('west', 'south', 'bottom', 'start'),
                                       ('east', 'north', 'top', 'end'),
                                       (dx, dy, dz, dt)):
            if axmin not in kwargs.keys(): kwargs[axmin] = 0
            if axmax not in kwargs.keys(): kwargs[axmax] = 0
            if min(kwargs[axmin],
                   kwargs[axmax]) == max(kwargs[axmin], kwargs[axmax]):
                kwargs[axmax] += delta

        spacebins = lambda a, b, delta: arange(
            min(a, b) - (min(a, b) % (delta * 1)),
            max(a, b) - (max(a, b) % (delta * -1)), delta)

        for x in spacebins(kwargs['west'], kwargs['east'], dx):

            for y in spacebins(kwargs['south'], kwargs['north'], dy):

                for z in spacebins(kwargs['top'], kwargs['bottom'], dz):

                    for t in arange(kwargs['start'].date(), kwargs['end'],
                                    dt).astype(datetime):

                        yield dict(
                            zip((
                                'west',
                                'east',
                                'south',
                                'north',
                                'top',
                                'bottom',
                                'start',
                                'end',
                            ), (
                                x,
                                x + dx,
                                y,
                                y + dy,
                                z,
                                z + dz,
                                t,
                                t + dt,
                            )))


class parallelindex(index):
    """ run index jobs in a parallel processing pool """

    def __call__(self, *, callback, **passkwargs):
        assert len(
            self.kwargslist) > 1, 'nothing to parallelize when bins=False'
        with Pool(self.pool) as p:
            return list(
                p.map(
                    self.__call_generator__,
                    zip((callback for _ in self.kwargslist), self.kwargslist,
                        (passkwargs for _ in self.kwargslist))))

    def __call_generator__(self, args):
        callback, kwargs, passkwargs = args
        #seed=f'{callback.__module__}.{callback.__name__}:{json.dumps(passkwargs, default=str, sort_keys=True)}'
        seed = self.hash_seed(callback, passkwargs)
        if not self.serialized(kwargs, seed):
            self.insert_hash(kwargs, seed, callback(**passkwargs, **kwargs))
        elif self.inmemory:
            self.update_hash(kwargs, seed, callback(**passkwargs, **kwargs))
        return pickle.loads(self.serialized(kwargs, seed))


'''
class memindex(index):
    pass
'''
