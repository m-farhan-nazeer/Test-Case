"""
Basic tests for the document processing system.
"""

import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_word_txt.processor import DocumentProcessor
from pdf_word_txt.config import ProcessingConfig
from pdf_word_txt.utils import clean_text, generate_summary, extract_metadata


def test_text_processing():
    """Test basic text processing utilities."""
    print("Testing text processing utilities...")
    
    # Test text cleaning
    dirty_text = "  This   is\n\n\n  some   messy\r\n\r\ntext  \t\t  "
    clean = clean_text(dirty_text)
    expected = "This is\n\nsome messy\n\ntext"
    assert clean == expected, f"Expected '{expected}', got '{clean}'"
    print("✓ Text cleaning works")
    
    # Test summary generation
    long_text = "This is the first sentence. " * 50
    summary = generate_summary(long_text, max_length=100)
    assert len(summary) <= 100, f"Summary too long: {len(summary)} chars"
    assert summary.endswith('.'), "Summary should end with period"
    print("✓ Summary generation works")


def test_txt_processing():
    """Test TXT file processing."""
    print("Testing TXT file processing...")
    
    # Create temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("This is a test document.\n\nIt has multiple paragraphs.\n\nAnd some content.")
        temp_file = f.name
    
    try:
        processor = DocumentProcessor()
        result = processor.process_file(temp_file)
        
        assert result['format'] == 'txt', f"Expected format 'txt', got '{result['format']}'"
        assert 'This is a test document' in result['content'], "Content not extracted properly"
        assert result['metadata']['word_count'] > 0, "Word count should be positive"
        assert result['metadata']['line_count'] > 0, "Line count should be positive"
        
        print("✓ TXT processing works")
        
    finally:
        os.unlink(temp_file)


def test_config():
    """Test configuration system."""
    print("Testing configuration...")
    
    config = ProcessingConfig()
    assert config.generate_summary == True, "Default summary generation should be True"
    assert config.max_workers > 0, "Max workers should be positive"
    assert config.summary_max_length > 0, "Summary max length should be positive"
    
    # Test validation
    try:
        bad_config = ProcessingConfig(summary_max_length=-1)
        bad_config.validate()
        assert False, "Should have raised ValueError for negative summary_max_length"
    except ValueError:
        pass  # Expected
    
    print("✓ Configuration works")


def test_metadata_extraction():
    """Test metadata extraction."""
    print("Testing metadata extraction...")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content")
        temp_file = Path(f.name)
    
    try:
        metadata = extract_metadata(temp_file)
        
        assert 'filename' in metadata, "Filename should be in metadata"
        assert 'file_size_bytes' in metadata, "File size should be in metadata"
        assert 'created_date' in metadata, "Created date should be in metadata"
        assert metadata['file_extension'] == '.txt', "File extension should be .txt"
        
        print("✓ Metadata extraction works")
        
    finally:
        temp_file.unlink()


def run_basic_tests():
    """Run all basic tests."""
    print("Running basic tests for PDF/Word/TXT processor...")
    print("=" * 50)
    
    try:
        test_text_processing()
        test_config()
        test_metadata_extraction()
        test_txt_processing()
        
        print("=" * 50)
        print("✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
