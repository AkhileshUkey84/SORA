#!/usr/bin/env python3
"""
Demo Configuration for Hackathon
Optimizes system settings for reliable demo performance with intelligent fallbacks.
"""

import os
import json
from pathlib import Path

def setup_demo_environment():
    """Configure environment variables for optimal demo performance"""
    
    # Demo mode for enhanced fallback behavior
    os.environ['DEMO_MODE'] = 'true'
    
    # Rate limiting - more conservative for demo
    os.environ['LLM_RATE_LIMIT_INTERVAL'] = '15'  # 15 seconds between requests (safer)
    
    # Enable comprehensive logging
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['STRUCTURED_LOGGING'] = 'true'
    
    # Database settings optimized for demo
    os.environ['MAX_QUERY_ROWS'] = '100'
    os.environ['ENABLE_ROW_LIMITS'] = 'true'
    
    # Enhanced security for demo
    os.environ['SECURITY_MODE'] = 'educational'  # Show helpful error messages
    
    print("✅ Demo environment configured successfully!")

def verify_demo_setup():
    """Verify all demo components are working"""
    
    print("🔍 Verifying demo setup...")
    
    # Check required files
    required_files = [
        'main.py',
        'services/llm_service.py',
        'agents/sql_generator.py',
        'agents/audit_agent.py',
        'routes/intelligence.py'
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing: {file}")
            return False
        else:
            print(f"✅ Found: {file}")
    
    # Check uploads directory
    uploads_dir = Path('./uploads')
    if not uploads_dir.exists():
        uploads_dir.mkdir(exist_ok=True)
        print("📁 Created uploads directory")
    
    parquet_files = list(uploads_dir.glob('*.parquet'))
    print(f"📊 Found {len(parquet_files)} demo datasets")
    
    print("✅ Demo setup verification complete!")
    return True

def create_demo_datasets_info():
    """Create information about available demo datasets"""
    
    uploads_dir = Path('./uploads')
    if not uploads_dir.exists():
        print("❌ No uploads directory found")
        return
    
    datasets = []
    for file in uploads_dir.glob('*.parquet'):
        try:
            import pandas as pd
            df = pd.read_parquet(file)
            
            datasets.append({
                'filename': file.name,
                'name': file.stem,
                'rows': len(df),
                'columns': list(df.columns)[:10],  # First 10 columns
                'total_columns': len(df.columns),
                'sample_questions': generate_sample_questions(file.stem, df.columns)
            })
        except Exception as e:
            print(f"⚠️ Could not analyze {file.name}: {e}")
    
    # Save dataset info for easy reference
    info_file = Path('./demo_datasets.json')
    with open(info_file, 'w') as f:
        json.dump(datasets, f, indent=2)
    
    print(f"📋 Created demo dataset info: {info_file}")
    return datasets

def generate_sample_questions(dataset_name, columns):
    """Generate sample questions for each dataset based on column names"""
    
    questions = []
    column_names = [col.lower() for col in columns]
    
    # Generic questions
    questions.append("Show me a summary of this data")
    questions.append("What are the key patterns in this dataset?")
    
    # Date-based questions
    date_cols = [col for col in column_names if any(word in col for word in ['date', 'time', 'created'])]
    if date_cols:
        questions.append(f"Show me trends over time")
        questions.append("Which day of the week has the most activity?")
    
    # Categorical questions
    cat_cols = [col for col in column_names if any(word in col for word in ['type', 'category', 'status', 'industry', 'department'])]
    if cat_cols:
        questions.append(f"Break down the data by {cat_cols[0]}")
        questions.append("Which category performs best?")
    
    # Numeric questions
    num_cols = [col for col in column_names if any(word in col for word in ['amount', 'value', 'price', 'sales', 'revenue', 'count'])]
    if num_cols:
        questions.append("What are the top values?")
        questions.append("Show me the distribution of amounts")
    
    return questions[:6]  # Limit to 6 sample questions

def print_demo_commands():
    """Print helpful commands for running the demo"""
    
    print("\n🚀 HACKATHON DEMO COMMANDS")
    print("=" * 50)
    print("1. Start the server:")
    print("   python main.py")
    print()
    print("2. Test basic endpoint:")
    print("   curl http://localhost:8000/")
    print()
    print("3. Test intelligence dashboard:")
    print("   curl http://localhost:8000/api/v1/intelligence/dashboard")
    print()
    print("4. Demo-friendly queries to try:")
    
    # Load dataset info if available
    info_file = Path('./demo_datasets.json')
    if info_file.exists():
        with open(info_file) as f:
            datasets = json.load(f)
        
        for dataset in datasets[:2]:  # Show first 2 datasets
            print(f"\n   📊 Dataset: {dataset['name']} ({dataset['rows']:,} rows)")
            for question in dataset['sample_questions'][:3]:
                print(f"   • {question}")
    
    print("\n🎯 HACKATHON DIFFERENTIATORS TO HIGHLIGHT:")
    print("✨ Real-time AI reasoning transparency")
    print("🔍 Self-auditing with counterfactual validation")
    print("📚 Educational security (teaches instead of blocks)")
    print("🧠 Proactive insight discovery")
    print("💡 Smart follow-up question suggestions")

def main():
    """Main demo configuration function"""
    
    print("🎪 HACKATHON DEMO CONFIGURATION")
    print("=" * 50)
    
    # Setup environment
    setup_demo_environment()
    
    # Verify setup
    if not verify_demo_setup():
        print("❌ Demo setup verification failed!")
        return
    
    # Analyze available datasets
    datasets = create_demo_datasets_info()
    if datasets:
        print(f"✅ Configured {len(datasets)} demo datasets")
    
    # Print demo commands
    print_demo_commands()
    
    print("\n🎉 DEMO READY FOR HACKATHON!")
    print("The system is optimized for:")
    print("• Reliable fallback behavior when rate-limited")
    print("• User-friendly error messages")
    print("• Educational security guidance")
    print("• Comprehensive reasoning transparency")
    print()
    print("💡 TIP: Set GEMINI_API_KEY environment variable before starting")

if __name__ == "__main__":
    main()
