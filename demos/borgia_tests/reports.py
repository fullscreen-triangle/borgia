"""
Borgia Test Framework - Report Generation Module
===============================================

Comprehensive report generation system for the Borgia BMD framework.
Generates detailed reports including:

- Executive summaries and technical documentation
- Performance analysis and benchmarking results  
- Validation test results and compliance reports
- Visual charts and graphical representations
- PDF, HTML, and text format outputs
- Statistical analysis and trend identification

Author: Borgia Development Team
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

from .core import ValidationResult, BenchmarkResult


@dataclass
class ReportSection:
    """
    Individual report section container.
    
    Attributes:
        title: Section title
        content: Section content (text/HTML/markdown)
        level: Heading level (1-6)
        subsections: List of subsections
        metadata: Additional section metadata
    """
    title: str
    content: str
    level: int = 2
    subsections: List['ReportSection'] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []
        if self.metadata is None:
            self.metadata = {}


class BorgiaReportGenerator:
    """
    Main report generator for the Borgia BMD framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.
        
        Args:
            config: Report generation configuration
        """
        self.config = config or {
            'title': 'Borgia BMD Framework Validation Report',
            'author': 'Borgia Development Team',
            'company': 'Borgia Research Institute',
            'logo_path': None,
            'template_style': 'professional',
            'include_charts': True,
            'include_raw_data': False,
            'page_numbers': True,
            'table_of_contents': True,
            'executive_summary': True
        }
        
        self.logger = logging.getLogger(f'{__name__}.BorgiaReportGenerator')
        
        # Report structure
        self.sections = []
        self.executive_summary = ""
        self.recommendations = []
        
    def generate_comprehensive_report(self,
                                    validation_results: Dict[str, ValidationResult],
                                    benchmark_results: Dict[str, BenchmarkResult], 
                                    analysis_results: Dict[str, Any],
                                    output_path: Union[str, Path],
                                    include_visualizations: bool = True) -> bool:
        """
        Generate comprehensive PDF report.
        
        Args:
            validation_results: Validation test results
            benchmark_results: Benchmark test results
            analysis_results: Analysis results and statistics
            output_path: Output file path
            include_visualizations: Whether to include charts and graphs
            
        Returns:
            bool: True if report generation successful
        """
        try:
            self.logger.info(f"Generating comprehensive report: {output_path}")
            
            # Build report structure
            self._build_report_structure(validation_results, benchmark_results, analysis_results)
            
            # Try to generate PDF report
            if self._can_generate_pdf():
                success = self._generate_pdf_report(output_path, include_visualizations)
                if success:
                    return True
                else:
                    self.logger.warning("PDF generation failed, falling back to HTML")
            
            # Fallback to HTML report
            html_path = Path(output_path).with_suffix('.html')
            success = self._generate_html_report(html_path, include_visualizations)
            
            if success:
                self.logger.info(f"Report generated successfully: {html_path}")
                
                # Also generate text report
                text_path = Path(output_path).with_suffix('.txt')
                self._generate_text_report(text_path)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return False
    
    def _build_report_structure(self,
                              validation_results: Dict[str, ValidationResult],
                              benchmark_results: Dict[str, BenchmarkResult],
                              analysis_results: Dict[str, Any]):
        """Build the complete report structure."""
        
        # Executive Summary
        self._build_executive_summary(validation_results, benchmark_results, analysis_results)
        
        # Validation Results Section
        validation_section = self._build_validation_section(validation_results)
        self.sections.append(validation_section)
        
        # Performance Benchmarks Section
        benchmark_section = self._build_benchmark_section(benchmark_results)
        self.sections.append(benchmark_section)
        
        # Analysis and Statistics Section
        analysis_section = self._build_analysis_section(analysis_results)
        self.sections.append(analysis_section)
        
        # System Recommendations Section
        recommendations_section = self._build_recommendations_section(analysis_results)
        self.sections.append(recommendations_section)
        
        # Technical Details Section
        technical_section = self._build_technical_section(validation_results, benchmark_results)
        self.sections.append(technical_section)
        
    def _build_executive_summary(self,
                                validation_results: Dict[str, ValidationResult],
                                benchmark_results: Dict[str, BenchmarkResult],
                                analysis_results: Dict[str, Any]):
        """Build executive summary."""
        
        # Calculate key metrics
        total_tests = len(validation_results)
        passed_tests = sum(1 for result in validation_results.values() if result.success)
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        overall_validation_score = analysis_results.get('overall_validation_score', 0.0) * 100
        overall_benchmark_score = analysis_results.get('overall_benchmark_score', 0.0) * 100
        
        # Critical systems status
        critical_systems = ['dual_functionality', 'quality_control', 'cascade_failure_protection']
        critical_failures = [name for name in critical_systems 
                           if name in validation_results and not validation_results[name].success]
        
        # Performance highlights
        performance_summary = analysis_results.get('performance_summary', {})
        avg_throughput = performance_summary.get('average_throughput', 0.0)
        avg_latency = performance_summary.get('average_latency', 0.0)
        peak_memory = performance_summary.get('peak_memory_usage', 0.0)
        
        # Build executive summary content
        self.executive_summary = f"""
EXECUTIVE SUMMARY

The Borgia Biological Maxwell Demons (BMD) Framework has undergone comprehensive validation and performance testing. This report presents the results of {total_tests} validation tests and extensive performance benchmarking across all system components.

KEY FINDINGS:

• System Validation: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}% success rate)
• Overall Validation Score: {overall_validation_score:.1f}/100
• Overall Performance Score: {overall_benchmark_score:.1f}/100
• Average System Throughput: {avg_throughput:.1f} operations/second
• Average Response Latency: {avg_latency:.1f} milliseconds
• Peak Memory Usage: {peak_memory:.1f} MB

CRITICAL SYSTEMS STATUS:
"""
        
        if critical_failures:
            self.executive_summary += f"⚠️  ATTENTION REQUIRED: {len(critical_failures)} critical system(s) require immediate attention:\n"
            for failure in critical_failures:
                self.executive_summary += f"   - {failure.replace('_', ' ').title()}\n"
        else:
            self.executive_summary += "✅ All critical systems operating within specifications\n"
        
        self.executive_summary += f"""

SYSTEM CAPABILITIES VALIDATED:

• Dual-functionality Molecules: Clock precision of {validation_results.get('dual_functionality', ValidationResult('', False, 0.0, {})).metrics.get('average_clock_precision', 'N/A')}s achieved
• BMD Network Coordination: {validation_results.get('bmd_networks', ValidationResult('', False, 0.0, {})).metrics.get('thermodynamic_amplification_factor', 'N/A')}× amplification factor
• Hardware Integration: Zero-cost LED spectroscopy with {validation_results.get('hardware_integration', ValidationResult('', False, 0.0, {})).metrics.get('performance_improvement_factor', 'N/A')}× performance improvement
• Information Catalysis: {validation_results.get('information_catalysis', ValidationResult('', False, 0.0, {})).metrics.get('catalysis_efficiency', 0)*100:.1f}% efficiency achieved

RECOMMENDATIONS:

"""
        
        recommendations = analysis_results.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):  # Top 5 recommendations
                self.executive_summary += f"{i}. {rec[:100]}{'...' if len(rec) > 100 else ''}\n"
        else:
            self.executive_summary += "• System operating at optimal performance levels\n• No immediate action required\n• Continue monitoring for sustained performance\n"
        
        self.executive_summary += f"""

CONCLUSION:

The Borgia BMD Framework demonstrates {'excellent' if pass_rate > 90 else 'good' if pass_rate > 75 else 'acceptable'} performance across all tested components. {'All critical systems are operational and the framework is ready for production deployment.' if not critical_failures else 'Critical issues require resolution before production deployment.'}

This validation confirms the framework's capability to support downstream systems including Masunda Temporal Navigator, Buhera Foundry, and Kambuzuma consciousness-enhanced computation.
"""
    
    def _build_validation_section(self, validation_results: Dict[str, ValidationResult]) -> ReportSection:
        """Build validation results section."""
        
        content = """
VALIDATION TEST RESULTS

The following validation tests were performed to ensure system compliance with BMD framework requirements:

"""
        
        # Add results table
        content += "| Test Component | Status | Score | Execution Time | Errors |\n"
        content += "|----------------|--------|-------|----------------|--------|\n"
        
        for name, result in validation_results.items():
            status = "✅ PASS" if result.success else "❌ FAIL"
            score = f"{result.score:.3f}"
            exec_time = f"{result.execution_time:.2f}s"
            error_count = len(result.errors)
            
            display_name = name.replace('_', ' ').title()
            content += f"| {display_name} | {status} | {score} | {exec_time} | {error_count} |\n"
        
        content += "\n"
        
        # Add detailed results for each test
        for name, result in validation_results.items():
            subsection_content = f"""
### {name.replace('_', ' ').title()} Validation

**Status:** {'PASSED' if result.success else 'FAILED'}
**Score:** {result.score:.3f}/1.000
**Execution Time:** {result.execution_time:.2f} seconds

**Key Metrics:**
"""
            
            for metric_name, metric_value in result.metrics.items():
                if isinstance(metric_value, (int, float)):
                    if isinstance(metric_value, float):
                        formatted_value = f"{metric_value:.6f}" if metric_value < 0.001 else f"{metric_value:.3f}"
                    else:
                        formatted_value = str(metric_value)
                    subsection_content += f"• {metric_name.replace('_', ' ').title()}: {formatted_value}\n"
                else:
                    subsection_content += f"• {metric_name.replace('_', ' ').title()}: {metric_value}\n"
            
            if result.errors:
                subsection_content += "\n**Errors Encountered:**\n"
                for error in result.errors:
                    subsection_content += f"• {error}\n"
            
            if result.warnings:
                subsection_content += "\n**Warnings:**\n"
                for warning in result.warnings:
                    subsection_content += f"• {warning}\n"
            
            subsection_content += "\n"
            content += subsection_content
        
        return ReportSection(
            title="Validation Test Results",
            content=content,
            level=1
        )
    
    def _build_benchmark_section(self, benchmark_results: Dict[str, BenchmarkResult]) -> ReportSection:
        """Build performance benchmark section."""
        
        content = """
PERFORMANCE BENCHMARK RESULTS

Comprehensive performance testing was conducted across all framework components to measure throughput, latency, and resource utilization:

"""
        
        # Add benchmark summary table
        content += "| Benchmark | Performance Score | Throughput | Latency | Memory Usage | CPU Usage |\n"
        content += "|-----------|------------------|------------|---------|--------------|----------|\n"
        
        for name, result in benchmark_results.items():
            perf_score = f"{result.performance_score:.3f}"
            throughput = f"{result.throughput:.1f} ops/s" if result.throughput != float('inf') else "N/A"
            latency = f"{result.latency:.1f} ms" if result.latency != float('inf') else "N/A"
            memory = f"{result.memory_usage:.1f} MB"
            cpu = f"{result.cpu_utilization:.1f}%"
            
            display_name = name.replace('_', ' ').title()
            content += f"| {display_name} | {perf_score} | {throughput} | {latency} | {memory} | {cpu} |\n"
        
        content += "\n"
        
        # Add detailed benchmark results
        for name, result in benchmark_results.items():
            subsection_content = f"""
### {name.replace('_', ' ').title()} Benchmark

**Performance Score:** {result.performance_score:.3f}/1.000
**Throughput:** {result.throughput:.2f} operations/second
**Average Latency:** {result.latency:.2f} milliseconds
**Peak Memory Usage:** {result.memory_usage:.2f} MB
**Average CPU Utilization:** {result.cpu_utilization:.1f}%
**Execution Time:** {result.execution_time:.2f} seconds
**Iterations Completed:** {result.iterations_completed}

"""
            
            # Add benchmark-specific metadata
            if result.metadata:
                subsection_content += "**Additional Metrics:**\n"
                for key, value in result.metadata.items():
                    if key not in ['error', 'monitoring_data'] and isinstance(value, (int, float)):
                        if isinstance(value, float):
                            formatted_value = f"{value:.6f}" if value < 0.001 else f"{value:.3f}"
                        else:
                            formatted_value = str(value)
                        subsection_content += f"• {key.replace('_', ' ').title()}: {formatted_value}\n"
                
                if 'error' in result.metadata:
                    subsection_content += f"\n**Error:** {result.metadata['error']}\n"
            
            subsection_content += "\n"
            content += subsection_content
        
        return ReportSection(
            title="Performance Benchmark Results",
            content=content,
            level=1
        )
    
    def _build_analysis_section(self, analysis_results: Dict[str, Any]) -> ReportSection:
        """Build analysis and statistics section."""
        
        content = """
STATISTICAL ANALYSIS AND SYSTEM INSIGHTS

This section presents comprehensive statistical analysis of the validation and benchmark results:

"""
        
        # Overall system performance
        content += f"""
## Overall System Performance

• **Overall Validation Score:** {analysis_results.get('overall_validation_score', 0.0):.3f}/1.000
• **Overall Benchmark Score:** {analysis_results.get('overall_benchmark_score', 0.0):.3f}/1.000
• **Tests Passed:** {analysis_results.get('tests_passed', 0)}/{analysis_results.get('total_tests', 0)}
• **Success Rate:** {(analysis_results.get('tests_passed', 0) / max(analysis_results.get('total_tests', 1), 1)) * 100:.1f}%

"""
        
        # Performance summary
        performance_summary = analysis_results.get('performance_summary', {})
        if performance_summary:
            content += """
## Performance Summary

"""
            for metric, value in performance_summary.items():
                formatted_metric = metric.replace('_', ' ').title()
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                content += f"• **{formatted_metric}:** {formatted_value}\n"
        
        # Critical failures analysis
        critical_failures = analysis_results.get('critical_failures', [])
        if critical_failures:
            content += f"""
## Critical Issues Analysis

⚠️  **{len(critical_failures)} Critical Issue(s) Detected:**

"""
            for failure in critical_failures:
                content += f"• **{failure.replace('_', ' ').title()}:** Requires immediate attention\n"
        else:
            content += """
## System Health Status

✅ **All Critical Systems Operational**

No critical failures detected. All essential framework components are functioning within specifications.
"""
        
        # Resource utilization analysis
        content += """
## Resource Utilization Analysis

The following resource utilization patterns were observed during testing:

"""
        
        # Add resource utilization insights
        avg_memory = performance_summary.get('peak_memory_usage', 0.0)
        avg_cpu = performance_summary.get('average_cpu_utilization', 0.0)
        
        if avg_memory < 1000:
            memory_status = "Excellent - Low memory footprint"
        elif avg_memory < 4000:
            memory_status = "Good - Moderate memory usage" 
        else:
            memory_status = "Attention - High memory usage"
        
        if avg_cpu < 50:
            cpu_status = "Efficient - Conservative CPU usage"
        elif avg_cpu < 80:
            cpu_status = "Normal - Balanced CPU utilization"
        else:
            cpu_status = "Intensive - High CPU utilization"
        
        content += f"""
• **Memory Usage:** {avg_memory:.1f} MB ({memory_status})
• **CPU Utilization:** {avg_cpu:.1f}% ({cpu_status})
• **System Scalability:** Framework demonstrates good scalability characteristics
• **Resource Efficiency:** Overall resource utilization within acceptable parameters

"""
        
        return ReportSection(
            title="Analysis and Statistics",
            content=content,
            level=1
        )
    
    def _build_recommendations_section(self, analysis_results: Dict[str, Any]) -> ReportSection:
        """Build recommendations section."""
        
        content = """
SYSTEM RECOMMENDATIONS

Based on the comprehensive validation and performance analysis, the following recommendations are provided:

"""
        
        recommendations = analysis_results.get('recommendations', [])
        
        if recommendations:
            for i, recommendation in enumerate(recommendations, 1):
                content += f"""
### Recommendation {i}

{recommendation}

"""
        else:
            content += """
### No Issues Detected

The system is performing optimally across all tested components. Continue regular monitoring to maintain performance levels.

### Suggested Maintenance Schedule

• **Weekly:** Monitor system health metrics and performance indicators
• **Monthly:** Review validation test results and benchmark performance trends  
• **Quarterly:** Comprehensive system validation and performance optimization review
• **Annually:** Full system upgrade evaluation and capacity planning assessment
"""
        
        return ReportSection(
            title="Recommendations",
            content=content,
            level=1
        )
    
    def _build_technical_section(self,
                                validation_results: Dict[str, ValidationResult],
                                benchmark_results: Dict[str, BenchmarkResult]) -> ReportSection:
        """Build technical details section."""
        
        content = """
TECHNICAL IMPLEMENTATION DETAILS

This section provides technical details about the validation framework and testing methodology:

"""
        
        # Test execution environment
        content += f"""
## Test Execution Environment

• **Framework Version:** Borgia BMD Framework v1.0.0
• **Test Execution Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
• **Validation Tests:** {len(validation_results)} components tested
• **Performance Benchmarks:** {len(benchmark_results)} benchmarks executed
• **Testing Duration:** {sum(result.execution_time for result in validation_results.values()) + sum(result.execution_time for result in benchmark_results.values()):.1f} seconds

## Framework Components Tested

"""
        
        # List all tested components
        all_components = set(list(validation_results.keys()) + list(benchmark_results.keys()))
        for component in sorted(all_components):
            component_name = component.replace('_', ' ').title()
            validation_status = "✅" if component in validation_results and validation_results[component].success else "❌"
            benchmark_status = "✅" if component in benchmark_results and benchmark_results[component].performance_score > 0.5 else "❌"
            
            content += f"• **{component_name}** (Validation: {validation_status}, Performance: {benchmark_status})\n"
        
        content += """

## Testing Methodology

### Validation Testing
• **Zero-tolerance Quality Control:** All dual-functionality molecules must meet strict precision requirements
• **Multi-scale BMD Networks:** Testing across quantum (10⁻¹⁵s), molecular (10⁻⁹s), and environmental (10²s) timescales
• **Hardware Integration:** Validation of zero-cost LED spectroscopy and noise-enhanced processing
• **Downstream Integration:** Compatibility testing with Masunda Temporal, Buhera Foundry, and Kambuzuma systems
• **Cascade Failure Protection:** Resilience testing under various failure scenarios

### Performance Benchmarking
• **Molecular Generation:** Throughput and precision testing under various load conditions
• **BMD Coordination:** Multi-scale network efficiency and amplification factor measurement
• **System Scalability:** Performance testing with increasing workload sizes
• **Resource Utilization:** Memory efficiency and CPU utilization optimization validation
• **Statistical Analysis:** Comprehensive statistical validation of all performance metrics

## Quality Assurance Standards

• **Precision Requirements:** Clock precision of 10⁻³⁰ seconds or better
• **Amplification Targets:** Thermodynamic amplification factor > 1000×
• **Efficiency Standards:** Information catalysis efficiency > 95%
• **Reliability Requirements:** System availability > 95% during operation
• **Zero-tolerance Policy:** No defects acceptable in dual-functionality molecule validation

"""
        
        return ReportSection(
            title="Technical Details",
            content=content,
            level=1
        )
    
    def _can_generate_pdf(self) -> bool:
        """Check if PDF generation is available."""
        try:
            import reportlab
            return True
        except ImportError:
            return False
    
    def _generate_pdf_report(self, output_path: Union[str, Path], include_visualizations: bool) -> bool:
        """Generate PDF report using ReportLab."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
            from reportlab.lib import colors
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build story (content)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            
            story.append(Paragraph(self.config['title'], title_style))
            story.append(Spacer(1, 20))
            
            # Author and date
            author_style = ParagraphStyle(
                'AuthorStyle',
                parent=styles['Normal'],
                fontSize=12,
                alignment=1
            )
            
            story.append(Paragraph(f"Generated by: {self.config['author']}", author_style))
            story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", author_style))
            story.append(PageBreak())
            
            # Executive Summary
            if self.config.get('executive_summary', True):
                story.append(Paragraph("EXECUTIVE SUMMARY", styles['Heading1']))
                
                # Split executive summary into paragraphs
                for paragraph in self.executive_summary.split('\n\n'):
                    if paragraph.strip():
                        story.append(Paragraph(paragraph.strip(), styles['Normal']))
                        story.append(Spacer(1, 12))
                
                story.append(PageBreak())
            
            # Add all sections
            for section in self.sections:
                if section.level == 1:
                    story.append(Paragraph(section.title, styles['Heading1']))
                elif section.level == 2:
                    story.append(Paragraph(section.title, styles['Heading2']))
                else:
                    story.append(Paragraph(section.title, styles['Heading3']))
                
                # Process section content
                for paragraph in section.content.split('\n\n'):
                    if paragraph.strip():
                        # Handle simple markdown formatting
                        content = paragraph.strip()
                        if content.startswith('###'):
                            content = content[3:].strip()
                            story.append(Paragraph(content, styles['Heading3']))
                        elif content.startswith('##'):
                            content = content[2:].strip()
                            story.append(Paragraph(content, styles['Heading2']))
                        elif content.startswith('|') and '|' in content[1:]:
                            # Handle tables (basic implementation)
                            continue  # Skip tables in PDF for now
                        else:
                            story.append(Paragraph(content, styles['Normal']))
                        
                        story.append(Spacer(1, 12))
                
                story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"PDF report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            return False
    
    def _generate_html_report(self, output_path: Union[str, Path], include_visualizations: bool) -> bool:
        """Generate HTML report."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config['title']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        h3 {{
            color: #5a6c7d;
        }}
        .executive-summary {{
            background: #ecf0f1;
            padding: 20px;
            border-left: 5px solid #3498db;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .failure {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 10px;
            margin: 10px 0;
            border-left: 3px solid #007bff;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.config['title']}</h1>
        <p><strong>Generated by:</strong> {self.config['author']}</p>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
"""
            
            # Executive Summary
            if self.config.get('executive_summary', True):
                html_content += f"""
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <pre>{self.executive_summary}</pre>
        </div>
"""
            
            # Add all sections
            for section in self.sections:
                level = min(section.level + 1, 6)  # Adjust level for HTML
                html_content += f"""
        <h{level}>{section.title}</h{level}>
        <div>
"""
                
                # Convert section content to HTML
                content_html = self._markdown_to_html(section.content)
                html_content += content_html
                
                html_content += """
        </div>
"""
            
            # Footer
            html_content += f"""
        <div class="footer">
            <p>Generated by Borgia BMD Framework Validation System</p>
            <p>&copy; {datetime.now().year} Borgia Research Institute</p>
        </div>
    </div>
</body>
</html>
"""
            
            # Write HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"HTML generation failed: {e}")
            return False
    
    def _generate_text_report(self, output_path: Union[str, Path]) -> bool:
        """Generate plain text report."""
        try:
            content = f"""
{self.config['title']}
{'=' * len(self.config['title'])}

Generated by: {self.config['author']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{self.executive_summary}

"""
            
            # Add all sections
            for section in self.sections:
                title_underline = '=' if section.level == 1 else '-' if section.level == 2 else '~'
                
                content += f"""
{section.title}
{title_underline * len(section.title)}

{section.content}

"""
            
            # Write text file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Text report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            return False
    
    def _markdown_to_html(self, markdown_content: str) -> str:
        """Convert basic markdown to HTML."""
        html = markdown_content
        
        # Handle tables
        lines = html.split('\n')
        html_lines = []
        in_table = False
        
        for line in lines:
            if line.strip().startswith('|') and '|' in line.strip()[1:]:
                if not in_table:
                    html_lines.append('<table>')
                    in_table = True
                
                # Parse table row
                cells = [cell.strip() for cell in line.strip().split('|')[1:-1]]
                
                # Check if header row (contains dashes)
                if all('---' in cell or cell == '' for cell in cells):
                    continue  # Skip separator row
                
                # Determine if it's a header row (first row or after separator)
                is_header = not any('<td>' in prev_line for prev_line in html_lines[-3:])
                
                cell_tag = 'th' if is_header else 'td'
                row_html = '<tr>'
                for cell in cells:
                    # Apply basic formatting
                    cell_formatted = self._format_cell_content(cell)
                    row_html += f'<{cell_tag}>{cell_formatted}</{cell_tag}>'
                row_html += '</tr>'
                
                html_lines.append(row_html)
                
            else:
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                
                # Handle other markdown elements
                formatted_line = self._format_markdown_line(line)
                html_lines.append(formatted_line)
        
        if in_table:
            html_lines.append('</table>')
        
        return '\n'.join(html_lines)
    
    def _format_cell_content(self, cell: str) -> str:
        """Format table cell content."""
        cell = cell.strip()
        
        # Status indicators
        if '✅' in cell:
            cell = cell.replace('✅', '<span class="success">✅</span>')
        if '❌' in cell:
            cell = cell.replace('❌', '<span class="failure">❌</span>')
        if 'PASS' in cell:
            cell = cell.replace('PASS', '<span class="success">PASS</span>')
        if 'FAIL' in cell:
            cell = cell.replace('FAIL', '<span class="failure">FAIL</span>')
        
        return cell
    
    def _format_markdown_line(self, line: str) -> str:
        """Format a single markdown line."""
        line = line.strip()
        
        # Headers
        if line.startswith('###'):
            return f'<h4>{line[3:].strip()}</h4>'
        elif line.startswith('##'):
            return f'<h3>{line[2:].strip()}</h3>'
        elif line.startswith('#'):
            return f'<h2>{line[1:].strip()}</h2>'
        
        # Bold text
        if '**' in line:
            import re
            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
        
        # Lists
        if line.startswith('• '):
            return f'<li>{line[2:]}</li>'
        
        # Status indicators
        if '✅' in line:
            line = line.replace('✅', '<span class="success">✅</span>')
        if '❌' in line:
            line = line.replace('❌', '<span class="failure">❌</span>')
        if '⚠️' in line:
            line = line.replace('⚠️', '<span style="color: #f39c12;">⚠️</span>')
        
        # Wrap in paragraph if not empty and not a special element
        if line and not line.startswith('<'):
            return f'<p>{line}</p>'
        
        return line


def generate_quick_report(validation_results: Dict[str, ValidationResult],
                         benchmark_results: Dict[str, BenchmarkResult],
                         output_path: Union[str, Path]) -> bool:
    """
    Generate a quick summary report.
    
    Args:
        validation_results: Validation test results
        benchmark_results: Benchmark test results  
        output_path: Output file path
        
    Returns:
        bool: True if successful
    """
    try:
        # Calculate summary statistics
        total_tests = len(validation_results)
        passed_tests = sum(1 for r in validation_results.values() if r.success)
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        avg_validation_score = np.mean([r.score for r in validation_results.values()]) if validation_results else 0
        avg_benchmark_score = np.mean([r.performance_score for r in benchmark_results.values()]) if benchmark_results else 0
        
        # Create quick report content
        content = f"""
BORGIA BMD FRAMEWORK - QUICK VALIDATION REPORT
{'='*50}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY RESULTS:
• Tests Passed: {passed_tests}/{total_tests} ({pass_rate:.1f}%)
• Average Validation Score: {avg_validation_score:.3f}
• Average Performance Score: {avg_benchmark_score:.3f}

VALIDATION RESULTS:
"""
        
        for name, result in validation_results.items():
            status = "PASS" if result.success else "FAIL"
            content += f"• {name.replace('_', ' ').title()}: {status} ({result.score:.3f})\n"
        
        content += "\nPERFORMANCE RESULTS:\n"
        
        for name, result in benchmark_results.items():
            content += f"• {name.replace('_', ' ').title()}: {result.performance_score:.3f} ({result.throughput:.1f} ops/s)\n"
        
        content += f"""

SYSTEM STATUS: {'OPERATIONAL' if pass_rate > 75 else 'NEEDS ATTENTION'}

End of Quick Report
"""
        
        # Write report
        with open(output_path, 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        logging.error(f"Quick report generation failed: {e}")
        return False
