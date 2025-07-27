#!/usr/bin/env python3
"""
Compare load test results between regular and Ray-enabled APIs.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, Any
import numpy as np

def load_results(filename: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def compare_results(regular_results: Dict[str, Any], ray_results: Dict[str, Any]):
    """Compare results between regular and Ray APIs."""
    print("🔍 COMPARISON RESULTS")
    print("=" * 60)
    
    # Calculate improvements
    throughput_improvement = ((ray_results['requests_per_second'] - regular_results['requests_per_second']) / 
                             regular_results['requests_per_second']) * 100
    
    response_time_improvement = ((regular_results['avg_response_time'] - ray_results['avg_response_time']) / 
                                regular_results['avg_response_time']) * 100
    
    print(f"📊 Throughput:")
    print(f"   Regular API: {regular_results['requests_per_second']:.2f} req/s")
    print(f"   Ray API: {ray_results['requests_per_second']:.2f} req/s")
    print(f"   Improvement: {throughput_improvement:+.2f}%")
    
    print(f"\n⏱️  Response Time:")
    print(f"   Regular API: {regular_results['avg_response_time']:.3f}s")
    print(f"   Ray API: {ray_results['avg_response_time']:.3f}s")
    print(f"   Improvement: {response_time_improvement:+.2f}%")
    
    print(f"\n📈 Success Rate:")
    print(f"   Regular API: {regular_results['success_rate']:.2f}%")
    print(f"   Ray API: {ray_results['success_rate']:.2f}%")
    
    print(f"\n🎯 Ray-specific Metrics:")
    if 'ray_workers_used' in ray_results:
        print(f"   Ray Workers Used: {ray_results['ray_workers_used']}")
        print(f"   Unique Ray Workers: {ray_results['unique_ray_workers']}")
    
    # Determine winner
    if throughput_improvement > 0 and response_time_improvement > 0:
        print(f"\n🏆 Ray API performs better!")
        print(f"   +{throughput_improvement:.2f}% throughput")
        print(f"   +{response_time_improvement:.2f}% faster response time")
    elif throughput_improvement < 0 and response_time_improvement < 0:
        print(f"\n🏆 Regular API performs better!")
        print(f"   {throughput_improvement:.2f}% throughput")
        print(f"   {response_time_improvement:.2f}% response time")
    else:
        print(f"\n🤔 Mixed results - depends on priority")

def create_comparison_chart(regular_results: Dict[str, Any], ray_results: Dict[str, Any], output_file: str = "load_test_comparison.png"):
    """Create comparison charts."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Throughput comparison
    apis = ['Regular API', 'Ray API']
    throughputs = [regular_results['requests_per_second'], ray_results['requests_per_second']]
    colors = ['#ff7f0e', '#1f77b4']
    
    bars1 = ax1.bar(apis, throughputs, color=colors, alpha=0.7)
    ax1.set_title('Throughput Comparison (req/s)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Requests per Second')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, throughputs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Response time comparison
    response_times = [regular_results['avg_response_time'], ray_results['avg_response_time']]
    bars2 = ax2.bar(apis, response_times, color=colors, alpha=0.7)
    ax2.set_title('Average Response Time (seconds)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Response Time (s)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, response_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Success rate comparison
    success_rates = [regular_results['success_rate'], ray_results['success_rate']]
    bars3 = ax3.bar(apis, success_rates, color=colors, alpha=0.7)
    ax3.set_title('Success Rate (%)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars3, success_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Percentile comparison
    p95_times = [regular_results['p95_response_time'], ray_results['p95_response_time']]
    p99_times = [regular_results['p99_response_time'], ray_results['p99_response_time']]
    
    x = np.arange(len(apis))
    width = 0.35
    
    bars4_1 = ax4.bar(x - width/2, p95_times, width, label='95th Percentile', color='#ff7f0e', alpha=0.7)
    bars4_2 = ax4.bar(x + width/2, p99_times, width, label='99th Percentile', color='#1f77b4', alpha=0.7)
    
    ax4.set_title('Response Time Percentiles (seconds)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Response Time (s)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(apis)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4_1, p95_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, value in zip(bars4_2, p99_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison chart saved to {output_file}")
    
    return output_file

def create_detailed_report(regular_results: Dict[str, Any], ray_results: Dict[str, Any], output_file: str = "load_test_report.md"):
    """Create a detailed markdown report."""
    report = f"""# Load Test Comparison Report

## Overview
This report compares the performance of the Regular API vs Ray-enabled API under concurrent load.

## Test Configuration
- **Regular API**: Standard FastAPI with synchronous processing
- **Ray API**: FastAPI with Ray-based distributed processing
- **Total Requests**: {regular_results['total_requests']}
- **Concurrent Users**: Tested with multiple concurrent users

## Performance Metrics

### Throughput
| Metric | Regular API | Ray API | Improvement |
|--------|-------------|---------|-------------|
| Requests/sec | {regular_results['requests_per_second']:.2f} | {ray_results['requests_per_second']:.2f} | {((ray_results['requests_per_second'] - regular_results['requests_per_second']) / regular_results['requests_per_second'] * 100):+.2f}% |

### Response Times
| Metric | Regular API | Ray API | Improvement |
|--------|-------------|---------|-------------|
| Average | {regular_results['avg_response_time']:.3f}s | {ray_results['avg_response_time']:.3f}s | {((regular_results['avg_response_time'] - ray_results['avg_response_time']) / regular_results['avg_response_time'] * 100):+.2f}% |
| Median | {regular_results['median_response_time']:.3f}s | {ray_results['median_response_time']:.3f}s | - |
| 95th Percentile | {regular_results['p95_response_time']:.3f}s | {ray_results['p95_response_time']:.3f}s | - |
| 99th Percentile | {regular_results['p99_response_time']:.3f}s | {ray_results['p99_response_time']:.3f}s | - |

### Reliability
| Metric | Regular API | Ray API |
|--------|-------------|---------|
| Success Rate | {regular_results['success_rate']:.2f}% | {ray_results['success_rate']:.2f}% |
| Failed Requests | {regular_results['failed_requests']} | {ray_results['failed_requests']} |

### Ray-specific Metrics
| Metric | Value |
|--------|-------|
| Ray Workers Used | {ray_results.get('ray_workers_used', 'N/A')} |
| Unique Ray Workers | {ray_results.get('unique_ray_workers', 'N/A')} |

## Analysis

### Throughput Analysis
- **Regular API**: {regular_results['requests_per_second']:.2f} requests per second
- **Ray API**: {ray_results['requests_per_second']:.2f} requests per second
- **Difference**: {(ray_results['requests_per_second'] - regular_results['requests_per_second']):+.2f} req/s

### Response Time Analysis
- **Regular API**: Average response time of {regular_results['avg_response_time']:.3f} seconds
- **Ray API**: Average response time of {ray_results['avg_response_time']:.3f} seconds
- **Difference**: {(ray_results['avg_response_time'] - regular_results['avg_response_time']):+.3f} seconds

## Conclusion

"""
    
    # Add conclusion based on results
    throughput_improvement = ((ray_results['requests_per_second'] - regular_results['requests_per_second']) / 
                             regular_results['requests_per_second']) * 100
    response_time_improvement = ((regular_results['avg_response_time'] - ray_results['avg_response_time']) / 
                                regular_results['avg_response_time']) * 100
    
    if throughput_improvement > 0 and response_time_improvement > 0:
        report += f"""The Ray-enabled API demonstrates superior performance:

✅ **Throughput Improvement**: {throughput_improvement:+.2f}% higher throughput
✅ **Response Time Improvement**: {response_time_improvement:+.2f}% faster response times
✅ **Scalability**: Better handling of concurrent requests through distributed processing

**Recommendation**: Use Ray-enabled API for production environments with high concurrent load.
"""
    elif throughput_improvement < 0 and response_time_improvement < 0:
        report += f"""The Regular API demonstrates better performance:

❌ **Throughput**: {throughput_improvement:.2f}% lower throughput with Ray
❌ **Response Time**: {response_time_improvement:.2f}% slower response times with Ray
⚠️ **Overhead**: Ray introduces overhead that outweighs benefits for this workload

**Recommendation**: Use Regular API for current load levels.
"""
    else:
        report += f"""Mixed results:

📊 **Throughput**: {throughput_improvement:+.2f}% change
📊 **Response Time**: {response_time_improvement:+.2f}% change

**Recommendation**: Consider specific requirements and test with actual production load.
"""
    
    report += f"""
## Test Details
- **Test Duration**: {regular_results['total_time']:.2f} seconds
- **Total Requests**: {regular_results['total_requests']}
- **Test Date**: Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*Report generated automatically by load test comparison tool*
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Detailed report saved to {output_file}")
    return output_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare load test results')
    parser.add_argument('--regular_file', type=str, required=True, help='Regular API results file')
    parser.add_argument('--ray_file', type=str, required=True, help='Ray API results file')
    parser.add_argument('--output_chart', type=str, default='load_test_comparison.png', help='Output chart filename')
    parser.add_argument('--output_report', type=str, default='load_test_report.md', help='Output report filename')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.regular_file).exists():
        print(f"❌ Regular results file not found: {args.regular_file}")
        return
    
    if not Path(args.ray_file).exists():
        print(f"❌ Ray results file not found: {args.ray_file}")
        return
    
    # Load results
    print("📊 Loading results...")
    regular_results = load_results(args.regular_file)
    ray_results = load_results(args.ray_file)
    
    # Compare results
    compare_results(regular_results, ray_results)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    create_comparison_chart(regular_results, ray_results, args.output_chart)
    create_detailed_report(regular_results, ray_results, args.output_report)
    
    print("\n✅ Comparison completed!")

if __name__ == "__main__":
    main() 