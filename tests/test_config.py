#!/usr/bin/env python3
"""
Quick test script to verify TradeMonkey configuration
"Trust, but verify" - Ronald Reagan (probably about crypto configs)
"""

import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test configuration loading"""
    print("🧪 Testing TradeMonkey Configuration...")
    print("=" * 50)
    
    try:
        # Test importing config
        from config.settings import config
        print("✅ Configuration module imported successfully")
        
        # Test validation
        if config.validate_config():
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return False
        
        # Print summary
        config.print_config_summary()
        
        # Test exchange config
        exchange_config = config.get_exchange_config()
        print(f"\n🐙 Exchange Configuration:")
        print(f"  API Key: {'*' * 20}...{exchange_config['apiKey'][-4:] if exchange_config['apiKey'] else 'NOT SET'}")
        print(f"  Rate Limit: {exchange_config['enableRateLimit']}")
        print(f"  Type: {exchange_config['options']['defaultType']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_requirements():
    """Test if all required packages are available"""
    print("\n📦 Testing Required Packages...")
    print("=" * 50)
    
    required_packages = [
        'ccxt',
        'pandas', 
        'numpy',
        'ta',
        'aiohttp',
        'python-dotenv',
        'coloredlogs'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All required packages are available!")
        return True


def test_files():
    """Test if all required files exist"""
    print("\n📁 Testing Required Files...")
    print("=" * 50)
    
    required_files = [
        'config/settings.py',
        'config/strategies.json',
        'bot.py',
        'main.py',
        '.env.example'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ All required files are present!")
        return True


def main():
    """Run all tests"""
    print("🐵 TradeMonkey Lite Configuration Test Suite")
    print("=" * 60)
    
    # Check if .env file exists
    if not Path('.env').exists():
        print("⚠️  WARNING: No .env file found!")
        print("   Copy .env.example to .env and add your API keys")
        print("   cp .env.example .env")
        print()
    
    tests = [
        ("Required Files", test_files),
        ("Required Packages", test_requirements), 
        ("Configuration", test_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n💥 {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! TradeMonkey is ready to rock! 🚀")
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Add your Kraken API keys to .env")
        print("3. Run: python main.py --paper")
        return 0
    else:
        print("😞 Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
