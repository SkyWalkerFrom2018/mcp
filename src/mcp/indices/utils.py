import logging
from pathlib import Path
from typing import Optional, Union, List, Callable, Dict
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from datetime import datetime
from itertools import chain
import hashlib
from llama_index.readers.file import PyMuPDFReader, DocxReader, MarkdownReader
from llama_index.core.readers import SimpleDirectoryReader

class FileLoader:
    """文件加载工具类"""
    
    @staticmethod
    def load_files(
        file_paths: Union[str, List[str], Path, List[Path]],
        file_extractor: Optional[Callable] = None,
        recursive: bool = True
    ) -> List[Document]:
        """加载多个文件（修复递归问题）"""
        
        def process_path(path: Path) -> List[Document]:
            # 处理单个路径（可能是文件或目录）
            if path.is_dir():
                return FileLoader.load_directory(
                    path, 
                    recursive=recursive,
                    file_extractor=file_extractor
                )
            else:
                return FileLoader._load_single_file(path, file_extractor)
        
        # 统一路径处理逻辑
        paths = [Path(p) for p in (file_paths if isinstance(file_paths, list) else [file_paths])]
        documents = list(chain.from_iterable(process_path(p) for p in paths))
        
        return documents

    @staticmethod
    def load_directory(
        directory: Union[str, Path],
        recursive: bool = True,
        **kwargs
    ) -> List[Document]:
        """增强版目录加载"""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"路径不是目录: {dir_path}")
            
        documents = []
        for item in dir_path.iterdir():
            if item.is_dir() and recursive:
                # 递归处理子目录
                documents.extend(
                    FileLoader.load_directory(item, recursive=True, **kwargs)
                )
            elif item.is_file():
                # 使用单文件加载方法
                try:
                    docs = FileLoader._load_single_file(item, **kwargs)
                    documents.extend(docs)
                except Exception as e:
                    logging.warning(f"加载文件 {item} 失败: {str(e)}")
                    
        return documents

    @staticmethod
    def _get_unified_extractor(custom_extractor: Optional[Dict] = None) -> Dict[str, Union[BaseReader, Callable]]:
        """返回解析器实例或可调用对象"""
        default = {
            ".pdf": PyMuPDFReader(),
            ".docx": DocxReader(),
            ".md": MarkdownReader(),
            ".txt": lambda path: SimpleDirectoryReader(input_files=[path]).load_data()
        }
        return {**default, **(custom_extractor or {})}

    @staticmethod
    def _get_file_metadata(path: Path) -> dict:
        """统一的元数据生成方法"""
        stat = path.stat()
        return {
            "file_size": stat.st_size,
            "creation_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_hash": hashlib.md5(path.read_bytes()).hexdigest()  # 新增文件哈希
        }

    @staticmethod
    def _load_single_file(
        path: Path,
        file_extractor: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """增强单文件加载"""
        # 获取文件类型对应的解析器
        ext = path.suffix.lower()
        extractors = FileLoader._get_unified_extractor(file_extractor)
        reader = extractors.get(ext)
        
        if not reader:
            raise ValueError(f"未注册的扩展名: {ext}")
            
        # 使用正确的解析器加载
        if isinstance(reader, BaseReader):
            return reader.load_data(file=path)
        else:
            # 处理函数类型的解析器
            return reader(str(path))

    @classmethod
    def load(
        cls,
        source: Union[str, Path, List[Union[str, Path, Document]]],
        **kwargs
    ) -> List[Document]:
        """修复参数传递问题"""
        is_directory = kwargs.pop('is_directory', None)
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            if is_directory is None:
                is_directory = path.is_dir()
                
            if is_directory:
                return cls.load_directory(path, **kwargs)
            else:
                return cls._load_single_file(path, **kwargs)
        elif isinstance(source, list):
            # 处理混合类型列表
            return list(chain.from_iterable(
                cls.load(item, **kwargs) for item in source
            ))
            
        if isinstance(source, Document):
            return [source]
        else:
            raise ValueError(f"无效的数据源: {source}")

