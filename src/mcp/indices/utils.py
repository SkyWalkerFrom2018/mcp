import os
import logging
from pathlib import Path
from typing import Union, List, Callable
from llama_index.core import Document
from llama_index.core.readers.download import download_loader

class FileLoader:
    """文件加载工具类"""
    
    @staticmethod
    def load_files(
        file_paths: Union[str, List[str], Path, List[Path]],
        file_extractor: Optional[Callable] = None
    ) -> List[Document]:
        """加载多个文件"""
        paths = [Path(p) for p in file_paths] if isinstance(file_paths, list) else [Path(file_paths)]
        documents = []
        
        for path in paths:
            if path.is_dir():
                documents.extend(FileLoader.load_directory(path))
            else:
                documents.extend(FileLoader._load_single_file(path, file_extractor))
                
        return documents

    @staticmethod
    def load_directory(
        directory: Union[str, Path],
        glob_pattern: str = "**/*.*",
        exclude_hidden: bool = True
    ) -> List[Document]:
        """加载目录文件"""
        dir_reader = download_loader("SimpleDirectoryReader")
        loader = dir_reader(
            str(directory),
            recursive=True,
            exclude_hidden=exclude_hidden
        )
        return loader.load_data()

    @staticmethod
    def _load_single_file(
        path: Path,
        file_extractor: Optional[Callable] = None
    ) -> List[Document]:
        """加载单个文件"""
        # ... 保持原有文件类型处理逻辑 ...
