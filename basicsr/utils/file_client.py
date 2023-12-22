# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py  # noqa: E501
from abc import ABCMeta, abstractmethod
import os
import cv2
import numpy as np
import random

import time
import sys
import imagesize

from multiprocessing import RawArray
import multiprocessing as mp
from basicsr.utils.dist_util import get_dist_info

from loguru import logger

class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class MemcachedBackend(BaseStorageBackend):
    """Memcached storage backend.

    Attributes:
        server_list_cfg (str): Config file for memcached server list.
        client_cfg (str): Config file for memcached client.
        sys_path (str | None): Additional path to be appended to `sys.path`.
            Default: None.
    """

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError('Please install memcached to enable MemcachedBackend.')

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(self.server_list_cfg, self.client_cfg)
        # mc.pyvector servers as a point which points to a memory cache
        self._mc_buffer = mc.pyvector()

    def get(self, filepath):
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf

class ShareDictBackend(BaseStorageBackend):
    """store the data in memory"""
    def __init__(self, imgdirs, store_undecoded=True, img_extensions=('.png', ), mp_count=8, channel=3, random_pick=None):
        super().__init__()
        self.imgdirs = imgdirs
        self.img_extensions = img_extensions
        self.store_undecoded = store_undecoded
        self.mp_count = mp_count
        self.channel = channel
        self.random_pick = random_pick

        if isinstance(imgdirs, str):
            imgdirs = [imgdirs]
        self._init_dict(imgdirs)

    @staticmethod
    def _init_sharedict_single(imgpaths, datas, store_undecoded=True):
        p = mp.current_process()
        for path_index, imgpath in enumerate(imgpaths):
            if path_index % 500 == 0:
                print(f'loading {path_index}/{len(imgpaths)} images into share dict in {p.name}')

            if imgpath not in datas: continue

            if store_undecoded is True:
                img = np.fromfile(imgpath, np.uint8)
            else:
                img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED).flatten()
            datas[imgpath][:] = img[:]

        print(f'loaded all {len(imgpaths)} images into share dict in {p.name}')

    def _init_dict(self, imgdirs): # read all imgs on initilization in case multi-processes
        self._dict = {}
        for imgdir_index, imgdir in enumerate(imgdirs):
            for filedir, _, filenames in os.walk(imgdir):
                for filename in filenames:
                    imgname, ext = os.path.splitext(filename)
                    if ext in self.img_extensions:
                        imgpath = os.path.join(filedir, filename)
                        if imgpath in self._dict: continue

                        if self.random_pick is not None:
                            # if random.randint(0, self.random_pick) == 0: continue
                            if random.random() > self.random_pick: continue

                        # we have to set the RawArray at the main process
                        if self.store_undecoded is True:
                            size = os.path.getsize(imgpath)
                        else:
                            w, h = imagesize.get(imgpath)
                            size = h*w*self.channel
                        self._dict[imgpath] = np.frombuffer(RawArray('B', size), dtype=np.uint8)

        imgpaths = list(self._dict.keys())
        print(f'Allocate all {len(imgpaths)} RawArray buffer')

        # load the data by multiprocessing
        length = len(imgpaths) // self.mp_count + 1
        jobs = []
        for index in range(self.mp_count):
            proc = mp.Process(target=ShareDictBackend._init_sharedict_single, args=(imgpaths[index*length:index*length+length], self._dict, self.store_undecoded))
            proc.start()
            jobs.append(proc)

        for proc in jobs:
            proc.join()
            proc.close()
        print(f'loaded all {len(imgpaths)} images into share dict')

        # decode all the data by multiprocessing
        if self.store_undecoded is False:
            for imgpath in self._dict:
                w, h = imagesize.get(imgpath)
                self._dict[imgpath] = self._dict[imgpath].reshape(h, w, self.channel)
            print(f'decoded all {len(imgpaths)} images')


    def get(self, filepath):
        filepath = str(filepath)
        if filepath not in self._dict:
            if self.store_undecoded is True:
                with open(filepath, 'rb') as f:
                    img = f.read()
            else:
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        else:
            img = self._dict[filepath]
        return img

    def get_text(self, filepath):
        filepath = str(filepath)
        if filepath not in self._dict:
            if self.store_undecoded is True:
                with open(filepath, 'rb') as f:
                    img = f.read()
            else:
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        else:
            img = self._dict[filepath].reshape(self.height, self.width, self.channel)
        return img

class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.

    Args:
        db_paths (str | list[str]): Lmdb database paths.
        client_keys (str | list[str]): Lmdb client keys. Default: 'default'.
        readonly (bool, optional): Lmdb environment parameter. If True,
            disallow any write operations. Default: True.
        lock (bool, optional): Lmdb environment parameter. If False, when
            concurrent access occurs, do not lock the database. Default: False.
        readahead (bool, optional): Lmdb environment parameter. If False,
            disable the OS filesystem readahead mechanism, which may improve
            random read performance when a database is larger than RAM.
            Default: False.

    Attributes:
        db_paths (list): Lmdb database path.
        _client (list): A list of several lmdb envs.
    """

    def __init__(self, db_paths, client_keys='default', readonly=True, lock=False, readahead=False, **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend.')

        if isinstance(client_keys, str):
            client_keys = [client_keys]

        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]
        assert len(client_keys) == len(self.db_paths), ('client_keys and db_paths should have the same length, '
                                                        f'but received {len(client_keys)} and {len(self.db_paths)}.')

        self._client = {}
        for client, path in zip(client_keys, self.db_paths):
            self._client[client] = lmdb.open(path, readonly=readonly, lock=lock, readahead=readahead, **kwargs)

    def get(self, filepath, client_key):
        """Get values according to the filepath from one lmdb named client_key.

        Args:
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
            client_key (str): Used for distinguishing different lmdb envs.
        """
        filepath = str(filepath)
        assert client_key in self._client, (f'client_key {client_key} is not in lmdb clients.')
        client = self._client[client_key]
        with client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class FileClient(object):
    """A general file client to access files in different backend.

    The client loads a file or text in a specified backend from its path
    and return it as a binary file. it can also register other backend
    accessor with a given name and backend class.

    Attributes:
        backend (str): The storage backend type. Options are "disk",
            "memcached" and "lmdb".
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        'disk': HardDiskBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
        'sharedict': ShareDictBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(f'Backend {backend} is not supported. Currently supported ones'
                             f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key='default'):
        # client_key is used only for lmdb, where different fileclients have
        # different lmdb environments.
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)
