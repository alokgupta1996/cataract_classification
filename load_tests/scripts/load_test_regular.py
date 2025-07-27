#!/usr/bin/env python3
"""
Load test script for regular FastAPI server (api.py)
Tests concurrent user performance without Ray.
"""

import asyncio
import aiohttp
import time
import json
import statistics
from pathlib import Path
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd

class LoadTester:
    """Load tester for API performance comparison."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, 
                          image_path: str = None, user_id: int = 0) -> Dict[str, Any]:
        """Make a single API request."""
        start_time = time.time()
        
        try:
            if image_path:
                # File upload request
                with open(image_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('image', f, filename=Path(image_path).name)
                    
                    async with session.post(f"{self.base_url}/{endpoint}", data=data) as response:
                        response_time = time.time() - start_time
                        result = await response.json()
                        
                        return {
                            'user_id': user_id,
                            'endpoint': endpoint,
                            'status_code': response.status,
                            'response_time': response_time,
                            'success': response.status == 200,
                            'error': None if response.status == 200 else result.get('detail', 'Unknown error')
                        }
            else:
                # Simple GET request
                async with session.get(f"{self.base_url}/{endpoint}") as response:
                    response_time = time.time() - start_time
                    result = await response.json()
                    
                    return {
                        'user_id': user_id,
                        'endpoint': endpoint,
                        'status_code': response.status,
                        'response_time': response_time,
                        'success': response.status == 200,
                        'error': None if response.status == 200 else result.get('detail', 'Unknown error')
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'user_id': user_id,
                'endpoint': endpoint,
                'status_code': 0,
                'response_time': response_time,
                'success': False,
                'error': str(e)
            }
    
    async def simulate_user(self, session: aiohttp.ClientSession, user_id: int, 
                           image_path: str, requests_per_user: int = 5) -> List[Dict[str, Any]]:
        """Simulate a single user making multiple requests."""
        user_results = []
        
        # Test different endpoints
        endpoints = ['predict_clip', 'predict_lgbm', 'predict_ensemble']
        
        for i in range(requests_per_user):
            endpoint = endpoints[i % len(endpoints)]
            result = await self.make_request(session, endpoint, image_path, user_id)
            user_results.append(result)
            
            # Small delay between requests
            await asyncio.sleep(0.1)
            
        return user_results
    
    async def run_load_test(self, num_users: int, requests_per_user: int, 
                           image_path: str, duration: int = 60) -> Dict[str, Any]:
        """Run the load test."""
        print(f"🚀 Starting load test with {num_users} users, {requests_per_user} requests per user")
        print(f"📊 Total requests: {num_users * requests_per_user}")
        print(f"⏱️  Duration: {duration} seconds")
        print("=" * 60)
        
        start_time = time.time()
        all_results = []
        
        async with aiohttp.ClientSession() as session:
            # Create tasks for all users
            tasks = []
            for user_id in range(num_users):
                task = self.simulate_user(session, user_id, image_path, requests_per_user)
                tasks.append(task)
            
            # Run all tasks concurrently
            user_results = await asyncio.gather(*tasks)
            
            # Flatten results
            for user_result in user_results:
                all_results.extend(user_result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        successful_requests = [r for r in all_results if r['success']]
        failed_requests = [r for r in all_results if not r['success']]
        
        response_times = [r['response_time'] for r in successful_requests]
        
        stats = {
            'total_requests': len(all_results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(all_results) * 100,
            'total_time': total_time,
            'requests_per_second': len(all_results) / total_time,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times) if response_times else 0,
            'p99_response_time': statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times) if response_times else 0,
            'results': all_results
        }
        
        return stats
    
    def print_results(self, stats: Dict[str, Any], api_type: str):
        """Print load test results."""
        print(f"\n📊 {api_type.upper()} API LOAD TEST RESULTS")
        print("=" * 60)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Success Rate: {stats['success_rate']:.2f}%")
        print(f"Total Time: {stats['total_time']:.2f}s")
        print(f"Throughput: {stats['requests_per_second']:.2f} req/s")
        print(f"Average Response Time: {stats['avg_response_time']:.3f}s")
        print(f"Median Response Time: {stats['median_response_time']:.3f}s")
        print(f"Min Response Time: {stats['min_response_time']:.3f}s")
        print(f"Max Response Time: {stats['max_response_time']:.3f}s")
        print(f"95th Percentile: {stats['p95_response_time']:.3f}s")
        print(f"99th Percentile: {stats['p99_response_time']:.3f}s")
        
        if stats['failed_requests'] > 0:
            print(f"\n❌ Failed Requests:")
            for result in stats['results']:
                if not result['success']:
                    print(f"  User {result['user_id']}, {result['endpoint']}: {result['error']}")
    
    def save_results(self, stats: Dict[str, Any], api_type: str):
        """Save results to file."""
        timestamp = int(time.time())
        filename = f"load_test_results_{api_type}_{timestamp}.json"
        
        # Remove detailed results for file size
        stats_to_save = {k: v for k, v in stats.items() if k != 'results'}
        
        with open(filename, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        return filename

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Load test for API performance')
    parser.add_argument('--num_users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--requests_per_user', type=int, default=5, help='Requests per user')
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"❌ Image not found: {args.image_path}")
        return
    
    # Run load test
    tester = LoadTester()
    stats = await tester.run_load_test(
        num_users=args.num_users,
        requests_per_user=args.requests_per_user,
        image_path=args.image_path,
        duration=args.duration
    )
    
    # Print and save results
    tester.print_results(stats, "regular")
    tester.save_results(stats, "regular")

if __name__ == "__main__":
    asyncio.run(main()) 