#!/usr/bin/env python3
"""
Comprehensive visualization generator for intrusion response research
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Generate all visualizations and analysis"""
    print("🎨 GENERATING COMPREHENSIVE RESEARCH VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # 1. Generate results visualizations
        from evaluation.visualizer import generate_visualizations
        generate_visualizations()
        
        # 2. Generate network visualizations
        from evaluation.network_visualizer import visualize_network
        visualize_network()
        
        # 3. Generate detailed analysis
        from evaluation.analyze_results import main as analyze_main
        analyze_main()
        
        print("\n" + "=" * 60)
        print("🎉 ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("📁 Check the 'results' directory for:")
        print("   📊 experiment_dashboard.png - Comprehensive results dashboard")
        print("   📈 publication_performance.png - Publication-quality performance plots")
        print("   📈 publication_analysis.png - Publication-quality analysis plots")
        print("   🌐 network_topology.png - Network structure visualization")
        print("   🔥 attack_progression.png - Attack propagation visualization")
        print("   📋 detailed_analysis.json - Detailed numerical analysis")
        print("\n📚 These visualizations are ready for your research paper!")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()