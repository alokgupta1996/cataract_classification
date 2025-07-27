#!/usr/bin/env python3
"""
Performance evaluation script for cataract classification models.
Tests inference speed, throughput, and accuracy metrics on test data.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference.clip_inference import CLIPInference
from src.inference.lgbm_inference import LGBMInference
from src.inference.batch_inference import BatchInference
from src.utils.logging_config import setup_project_logging, get_logger, log_performance_metrics, log_memory_usage, log_gpu_memory


def get_test_data_info(test_path: str) -> dict:
    """Get information about test data."""
    logger = get_logger(__name__)
    test_info = {}
    
    for category in ['normal', 'cataract']:
        category_path = os.path.join(test_path, category)
        if os.path.exists(category_path):
            image_files = list(Path(category_path).glob("*.jpg")) + list(Path(category_path).glob("*.png"))
            test_info[category] = {
                'path': category_path,
                'count': len(image_files),
                'files': image_files
            }
            logger.info(f"📁 {category}: {len(image_files)} images found")
        else:
            test_info[category] = {'path': category_path, 'count': 0, 'files': []}
            logger.warning(f"⚠️ {category} directory not found: {category_path}")
    
    total_images = sum(info['count'] for info in test_info.values())
    logger.info(f"📊 Total test images: {total_images}")
    
    return test_info


def evaluate_model_performance(model, test_info: dict, model_name: str, device: str = 'cpu') -> dict:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Model instance (CLIPInference, LGBMInference, or BatchInference)
        test_info (dict): Test data information
        model_name (str): Name of the model
        device (str): Device to run inference on
        
    Returns:
        dict: Performance metrics
    """
    logger = get_logger(__name__)
    logger.info(f"🔍 Evaluating {model_name} model performance...")
    
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    all_processing_times = []
    
    start_time = time.time()
    
    # Process each category
    for category, label in [('normal', 0), ('cataract', 1)]:
        if test_info[category]['count'] == 0:
            logger.warning(f"⚠️ No {category} images found, skipping...")
            continue
            
        logger.info(f"📊 Processing {category} images ({test_info[category]['count']} files)...")
        
        for i, image_path in enumerate(test_info[category]['files']):
            try:
                # Make prediction
                pred_start = time.time()
                result = model.predict_single(str(image_path))
                pred_time = time.time() - pred_start
                
                # Store results
                all_predictions.append(result['class'])
                all_true_labels.append(label)
                all_confidences.append(result['confidence'])
                all_processing_times.append(pred_time)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{test_info[category]['count']} {category} images")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {image_path}: {str(e)}")
                continue
    
    total_time = time.time() - start_time
    total_images = len(all_predictions)
    
    if total_images == 0:
        logger.error("❌ No images were processed successfully")
        return {}
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    
    # Calculate AUC (using confidence scores)
    try:
        auc = roc_auc_score(all_true_labels, all_confidences)
    except:
        auc = 0.0
    
    # Calculate throughput
    throughput = total_images / total_time if total_time > 0 else 0
    avg_processing_time = np.mean(all_processing_times) * 1000  # Convert to ms
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'total_images': total_images,
        'total_time': total_time,
        'throughput': throughput,
        'avg_processing_time_ms': avg_processing_time,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'confidences': all_confidences,
        'processing_times': all_processing_times
    }
    
    logger.info(f"✅ {model_name} evaluation completed:")
    logger.info(f"  📊 Accuracy: {accuracy:.4f}")
    logger.info(f"  📊 Precision: {precision:.4f}")
    logger.info(f"  📊 Recall: {recall:.4f}")
    logger.info(f"  📊 F1-Score: {f1:.4f}")
    logger.info(f"  📊 AUC: {auc:.4f}")
    logger.info(f"  ⏱️ Total time: {total_time:.2f}s")
    logger.info(f"  🚀 Throughput: {throughput:.2f} images/s")
    logger.info(f"  ⚡ Avg processing time: {avg_processing_time:.2f} ms")
    
    return metrics


def compare_models_performance(test_path: str = "processed_images/test") -> dict:
    """
    Compare performance of all models on test data.
    
    Args:
        test_path (str): Path to test data directory
        
    Returns:
        dict: Comparison results
    """
    logger = get_logger(__name__)
    logger.info("🔬 Starting model performance comparison...")
    
    # Get test data info
    test_info = get_test_data_info(test_path)
    
    if sum(info['count'] for info in test_info.values()) == 0:
        logger.error("❌ No test images found")
        return {}
    
    results = {}
    
    # Test CLIP model
    try:
        logger.info("🔄 Testing CLIP model...")
        clip_model = CLIPInference("models/clip/best_model.pth", device='cpu')
        clip_metrics = evaluate_model_performance(clip_model, test_info, "CLIP")
        results['CLIP'] = clip_metrics
    except Exception as e:
        logger.error(f"❌ CLIP model evaluation failed: {str(e)}")
        results['CLIP'] = {}
    
    # Test LGBM model
    try:
        logger.info("🔄 Testing LGBM model...")
        lgbm_model = LGBMInference("models/lgbm/trained_model.pkl", device='cpu')
        lgbm_metrics = evaluate_model_performance(lgbm_model, test_info, "LGBM")
        results['LGBM'] = lgbm_metrics
    except Exception as e:
        logger.error(f"❌ LGBM model evaluation failed: {str(e)}")
        results['LGBM'] = {}
    
    # Test Ensemble model
    try:
        logger.info("🔄 Testing Ensemble model...")
        ensemble_model = BatchInference(
            clip_model_path="models/clip/best_model.pth",
            lgbm_model_path="models/lgbm/trained_model.pkl",
            device='cpu'
        )
        ensemble_metrics = evaluate_model_performance(ensemble_model, test_info, "Ensemble")
        results['Ensemble'] = ensemble_metrics
    except Exception as e:
        logger.error(f"❌ Ensemble model evaluation failed: {str(e)}")
        results['Ensemble'] = {}
    
    # Create comparison summary
    logger.info("📊 Performance Comparison Summary:")
    logger.info("=" * 60)
    
    for model_name, metrics in results.items():
        if metrics:
            logger.info(f"🤖 {model_name}:")
            logger.info(f"  📊 Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"  📊 F1-Score: {metrics.get('f1_score', 0):.4f}")
            logger.info(f"  📊 AUC: {metrics.get('auc', 0):.4f}")
            logger.info(f"  ⏱️ Total time: {metrics.get('total_time', 0):.2f}s")
            logger.info(f"  🚀 Throughput: {metrics.get('throughput', 0):.2f} images/s")
            logger.info(f"  ⚡ Avg processing: {metrics.get('avg_processing_time_ms', 0):.2f} ms")
        else:
            logger.warning(f"⚠️ {model_name}: No metrics available")
    
    logger.info("=" * 60)
    
    # Save results
    save_performance_results(results, test_path)
    
    return results


def save_performance_results(results: dict, test_path: str):
    """Save performance results to files."""
    logger = get_logger(__name__)
    
    try:
        # Create results directory
        results_dir = Path("docs/performance")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save detailed results as JSON
        json_file = results_dir / f"performance_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, metrics in results.items():
            if metrics:
                json_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        json_metrics[key] = value.tolist()
                    elif isinstance(value, np.integer):
                        json_metrics[key] = int(value)
                    elif isinstance(value, np.floating):
                        json_metrics[key] = float(value)
                    else:
                        json_metrics[key] = value
                json_results[model_name] = json_metrics
        
        import json
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"💾 Detailed results saved to: {json_file}")
        
        # Create comparison plot
        create_performance_comparison_plot(results, results_dir, timestamp)
        
        # Create summary report
        create_performance_summary_report(results, results_dir, timestamp)
        
    except Exception as e:
        logger.error(f"❌ Failed to save performance results: {str(e)}")


def create_performance_comparison_plot(results: dict, results_dir: Path, timestamp: int):
    """Create performance comparison visualization."""
    logger = get_logger(__name__)
    
    try:
        # Prepare data for plotting
        models = []
        accuracies = []
        f1_scores = []
        auc_scores = []
        throughputs = []
        
        for model_name, metrics in results.items():
            if metrics:
                models.append(model_name)
                accuracies.append(metrics.get('accuracy', 0))
                f1_scores.append(metrics.get('f1_score', 0))
                auc_scores.append(metrics.get('auc', 0))
                throughputs.append(metrics.get('throughput', 0))
        
        if not models:
            logger.warning("⚠️ No valid metrics for plotting")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1-Score comparison
        bars2 = ax2.bar(models, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('F1-Score Comparison')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        # AUC comparison
        bars3 = ax3.bar(models, auc_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('AUC Comparison')
        ax3.set_ylabel('AUC')
        ax3.set_ylim(0, 1)
        for bar, auc in zip(bars3, auc_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # Throughput comparison
        bars4 = ax4.bar(models, throughputs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_title('Throughput Comparison')
        ax4.set_ylabel('Images/Second')
        for bar, tput in zip(bars4, throughputs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{tput:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = results_dir / f"performance_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Performance comparison plot saved to: {plot_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create performance comparison plot: {str(e)}")


def create_performance_summary_report(results: dict, results_dir: Path, timestamp: int):
    """Create a summary report of performance results."""
    logger = get_logger(__name__)
    
    try:
        report_file = results_dir / f"performance_summary_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Model Performance Summary\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("| Model | Accuracy | F1-Score | AUC | Throughput (img/s) | Avg Time (ms) |\n")
            f.write("|-------|----------|----------|-----|-------------------|---------------|\n")
            
            for model_name, metrics in results.items():
                if metrics:
                    f.write(f"| {model_name} | {metrics.get('accuracy', 0):.4f} | "
                           f"{metrics.get('f1_score', 0):.4f} | {metrics.get('auc', 0):.4f} | "
                           f"{metrics.get('throughput', 0):.2f} | "
                           f"{metrics.get('avg_processing_time_ms', 0):.2f} |\n")
                else:
                    f.write(f"| {model_name} | N/A | N/A | N/A | N/A | N/A |\n")
            
            f.write("\n## Best Performing Model\n\n")
            
            # Find best model by F1-score
            best_model = None
            best_f1 = 0
            
            for model_name, metrics in results.items():
                if metrics and metrics.get('f1_score', 0) > best_f1:
                    best_f1 = metrics.get('f1_score', 0)
                    best_model = model_name
            
            if best_model:
                f.write(f"**Best Model:** {best_model} (F1-Score: {best_f1:.4f})\n\n")
            else:
                f.write("**Best Model:** Unable to determine\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("- Use the model with the highest F1-Score for balanced performance\n")
            f.write("- Consider throughput requirements for production deployment\n")
            f.write("- Monitor model performance over time for drift detection\n")
        
        logger.info(f"📄 Performance summary report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create performance summary report: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model Performance Evaluation")
    parser.add_argument('--test-path', default='processed_images/test',
                       help='Path to test data directory')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_project_logging(args.log_level)
    logger.info("🚀 Starting model performance evaluation...")
    
    try:
        # Check if test path exists
        if not os.path.exists(args.test_path):
            logger.error(f"❌ Test path does not exist: {args.test_path}")
            return
        
        # Run performance comparison
        results = compare_models_performance(args.test_path)
        
        if results:
            logger.info("✅ Performance evaluation completed successfully")
            log_memory_usage(logger)
            log_gpu_memory(logger)
        else:
            logger.error("❌ Performance evaluation failed")
            
    except KeyboardInterrupt:
        logger.info("⚠️ Performance evaluation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Performance evaluation failed: {str(e)}")
        import traceback
        logger.error(f"📋 Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()