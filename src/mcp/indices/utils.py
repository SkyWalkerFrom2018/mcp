from pathlib import Path
from typing import Optional, Union, List, Callable, Dict
from llama_index.core import Document
from llama_index.core.readers.download import download_loader
from llama_index.core.readers import SimpleDirectoryReader
from datetime import datetime
import logging
from itertools import chain
import hashlib

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
        **kwargs
    ) -> List[Document]:
        """增强版目录加载（统一处理逻辑）"""
        # 使用自定义文件提取器来统一处理逻辑
        kwargs["file_extractor"] = FileLoader._get_unified_extractor(kwargs.get("file_extractor"))
        dir_reader = download_loader("SimpleDirectoryReader")
        loader = dir_reader(
            input_dir=str(directory),
            **kwargs
        )
        docs = loader.load_data()
        
        # 补充元数据
        for doc in docs:
            path = Path(doc.metadata["file_path"])
            doc.metadata.update(FileLoader._get_file_metadata(path))
            
        return docs

    @staticmethod
    def _get_unified_extractor(custom_extractor: Optional[Callable]) -> Dict[str, Callable]:
        """生成统一文件提取器"""
        from llama_index.core import SimpleDirectoryReader
        
        # 内置提取器映射
        default_extractors = {
            ".pdf": FileLoader._load_single_file,
            ".docx": FileLoader._load_single_file,
            ".md": FileLoader._load_single_file,
            ".txt": FileLoader._load_single_file
        }
        
        # 合并自定义提取器
        return {
            **default_extractors,
            **(custom_extractor or {})
        }

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
        file_extractor: Optional[Callable] = None
    ) -> List[Document]:
        """加载单个文件（完整实现）"""
        
        try:
            # 优先使用自定义解析器
            if file_extractor:
                return file_extractor(str(path))
            
            # 获取文件元数据
            stat = path.stat()
            metadata = {
                "file_name": path.name,
                "file_type": path.suffix[1:].lower(),
                "file_size": stat.st_size,
                "creation_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_path": str(path.resolve())
            }
            
            # 分文件类型处理
            if path.suffix.lower() == '.pdf':
                from llama_index.readers.file import PyMuPDFReader
                return PyMuPDFReader().load_data(file_path=path, metadata=metadata)
                
            elif path.suffix.lower() in ['.doc', '.docx']:
                from llama_index.readers.file import DocxReader
                return DocxReader().load_data(file_path=path, metadata=metadata)
                
            elif path.suffix.lower() == '.md':
                with open(path, 'r', encoding='utf-8') as f:
                    return [Document(text=f.read(), metadata=metadata)]
                    
            elif path.suffix.lower() in ['.txt', '.csv', '.json']:
                loader = SimpleDirectoryReader(input_files=[str(path)])
                return loader.load_data(show_progress=True)
                
            else:  # 处理未知类型
                logging.warning(f"未支持的文件类型: {path.suffix}，尝试以文本读取")
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        return [Document(text=f.read(), metadata=metadata)]
                except UnicodeDecodeError:
                    logging.error(f"无法解码文件: {path}")
                    return []
                    
        except Exception as e:
            logging.error(f"加载文件 {path} 失败: {str(e)}")
            return []
