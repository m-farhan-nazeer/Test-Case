"""
Example usage of the PDF/Word/TXT document processing system.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_word_txt.processor import DocumentProcessor
from pdf_word_txt.batch_processor import BatchProcessor
from pdf_word_txt.config import ProcessingConfig
from inject.config import Config


def example_single_file():
    """Example: Process a single document file."""
    print("=== Single File Processing Example ===")
    
    # Create processor with default config
    processor = DocumentProcessor()
    
    # Example file path (replace with actual file)
    file_path = "example_documents/sample.pdf"
    
    if not os.path.exists(file_path):
        print(f"Example file not found: {file_path}")
        print("Please create an example_documents folder with sample files")
        return
    
    try:
        result = processor.process_file(file_path)
        
        print(f"Successfully processed: {result['title']}")
        print(f"Content length: {len(result['content'])} characters")
        print(f"Word count: {result['metadata']['word_count']}")
        print(f"Format: {result['format']}")
        
        if result['summary']:
            print(f"Summary: {result['summary'][:200]}...")
        
    except Exception as e:
        print(f"Error processing file: {e}")


def example_folder_processing():
    """Example: Process all documents in a folder."""
    print("\n=== Folder Processing Example ===")
    
    # Create batch processor with custom config
    config = ProcessingConfig(
        parallel_processing=True,
        max_workers=2,
        generate_summary=True,
        integrate_with_rag=False  # Disable for example
    )
    
    batch_processor = BatchProcessor(config)
    
    # Example folder path
    folder_path = "example_documents"
    
    if not os.path.exists(folder_path):
        print(f"Example folder not found: {folder_path}")
        print("Please create an example_documents folder with sample files")
        return
    
    try:
        results = batch_processor.process_folder(folder_path)
        
        print(f"Found {results['total_files_found']} files")
        print(f"Successfully processed: {results['total_processed']}")
        print(f"Errors: {results['total_errors']}")
        
        # Show statistics
        stats = batch_processor.get_processing_stats(results)
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Formats processed: {stats['formats_processed']}")
        
        # Show first few results
        for i, doc in enumerate(results['results'][:3]):
            print(f"\nDocument {i+1}: {doc['title']}")
            print(f"  Format: {doc['format']}")
            print(f"  Size: {doc['metadata']['word_count']} words")
        
    except Exception as e:
        print(f"Error processing folder: {e}")


def example_with_config_integration():
    """Example: Using the integrated configuration system."""
    print("\n=== Configuration Integration Example ===")
    
    # Load main application config
    app_config = Config()
    
    # Get document processor settings
    doc_config = app_config.document_processor_config
    print(f"Document processor config: {doc_config}")
    
    # Create processing config from app config
    processing_config = ProcessingConfig(
        generate_summary=doc_config.get('generate_summary', True),
        summary_max_length=doc_config.get('summary_max_length', 500),
        parallel_processing=doc_config.get('parallel_processing', True),
        max_workers=doc_config.get('max_workers', 4),
        integrate_with_rag=doc_config.get('integrate_with_rag', True),
        max_file_size_mb=doc_config.get('max_file_size_mb', 100)
    )
    
    print(f"Processing config created from app config:")
    print(f"  Generate summary: {processing_config.generate_summary}")
    print(f"  Max workers: {processing_config.max_workers}")
    print(f"  RAG integration: {processing_config.integrate_with_rag}")


def example_supported_formats():
    """Example: Check supported file formats."""
    print("\n=== Supported Formats Example ===")
    
    processor = DocumentProcessor()
    supported = processor.get_supported_extensions()
    
    print("Supported file formats:")
    for ext in supported:
        print(f"  {ext}")
    
    # Check specific formats
    app_config = Config()
    test_formats = ['.pdf', '.docx', '.txt', '.xlsx']  # .xlsx not supported
    
    print("\nFormat support check:")
    for fmt in test_formats:
        supported = app_config.is_document_format_supported(fmt)
        status = "✓" if supported else "✗"
        print(f"  {fmt}: {status}")


def main():
    """Run all examples."""
    print("PDF/Word/TXT Document Processing Examples")
    print("=" * 50)
    
    example_supported_formats()
    example_with_config_integration()
    example_single_file()
    example_folder_processing()
    
    print("\n" + "=" * 50)
    print("Examples complete!")
    print("\nTo use the CLI:")
    print("  python -m pdf_word_txt.cli process-file document.pdf")
    print("  python -m pdf_word_txt.cli process-folder /path/to/documents/")
    print("  python -m pdf_word_txt.cli list-supported")


if __name__ == "__main__":
    main()
