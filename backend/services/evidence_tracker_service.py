# services/evidence_tracker_service.py
"""
Evidence Tracker Service that maintains provenance links between insights and source data.
Enables evidence-backed narratives by tracking which data points support each claim.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import structlog

logger = structlog.get_logger()


@dataclass
class Evidence:
    """Represents evidence supporting a claim or insight"""
    evidence_id: str
    claim: str
    supporting_data: List[Dict[str, Any]]
    source_rows: List[int]  # Row indices in result set
    calculation_method: str
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EvidenceLink:
    """Links an insight to its supporting evidence"""
    insight_text: str
    evidence_ids: List[str]
    data_references: List[str]  # e.g., "row 5, column 'revenue'"
    confidence_score: float


class EvidenceTrackerService:
    """
    Tracks evidence supporting analytical claims and insights.
    This is a key differentiator - every insight is backed by traceable data.
    """
    
    def __init__(self):
        self.evidence_store: Dict[str, Evidence] = {}
        self.evidence_links: List[EvidenceLink] = []
        self.logger = logger.bind(component="EvidenceTrackerService")
    
    def create_evidence(self, claim: str, data: List[Dict[str, Any]], 
                       method: str, source_rows: Optional[List[int]] = None) -> str:
        """
        Create an evidence record for a claim.
        Returns evidence ID for reference.
        """
        # Generate unique ID based on claim and data
        evidence_id = self._generate_evidence_id(claim, data)
        
        # Extract relevant data points
        supporting_data = self._extract_supporting_data(data, claim)
        
        # Calculate confidence based on data quality
        confidence = self._calculate_evidence_confidence(supporting_data, method)
        
        # Store evidence
        evidence = Evidence(
            evidence_id=evidence_id,
            claim=claim,
            supporting_data=supporting_data,
            source_rows=source_rows or list(range(len(data))),
            calculation_method=method,
            confidence=confidence
        )
        
        self.evidence_store[evidence_id] = evidence
        self.logger.info(f"Created evidence {evidence_id} for claim: {claim[:50]}...")
        
        return evidence_id
    
    def link_insight_to_evidence(self, insight: str, evidence_ids: List[str], 
                               data_refs: Optional[List[str]] = None) -> EvidenceLink:
        """
        Create a link between an insight and its supporting evidence.
        """
        # Calculate combined confidence
        confidences = [self.evidence_store[eid].confidence 
                      for eid in evidence_ids if eid in self.evidence_store]
        combined_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        link = EvidenceLink(
            insight_text=insight,
            evidence_ids=evidence_ids,
            data_references=data_refs or [],
            confidence_score=combined_confidence
        )
        
        self.evidence_links.append(link)
        return link
    
    def track_aggregation_evidence(self, metric_name: str, aggregation_type: str,
                                 raw_data: List[Dict[str, Any]], 
                                 result_value: Any) -> str:
        """
        Track evidence for aggregation calculations (SUM, AVG, etc).
        """
        # Build claim
        claim = f"{aggregation_type} of {metric_name} = {result_value}"
        
        # Extract values used in calculation
        values = []
        source_rows = []
        for i, row in enumerate(raw_data):
            if metric_name in row and row[metric_name] is not None:
                values.append({
                    "row": i,
                    "value": row[metric_name],
                    "contributes_to": aggregation_type
                })
                source_rows.append(i)
        
        # Create detailed method description
        method = self._describe_aggregation_method(aggregation_type, len(values), result_value)
        
        return self.create_evidence(
            claim=claim,
            data=values,
            method=method,
            source_rows=source_rows
        )
    
    def track_comparison_evidence(self, comparison_type: str, 
                                entity1: str, value1: Any,
                                entity2: str, value2: Any,
                                source_data: List[Dict[str, Any]]) -> str:
        """
        Track evidence for comparisons between entities.
        """
        # Calculate comparison metrics
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            diff = value1 - value2
            pct_diff = (diff / value2 * 100) if value2 != 0 else float('inf')
            comparison_result = f"{entity1} is {abs(pct_diff):.1f}% {'higher' if diff > 0 else 'lower'} than {entity2}"
        else:
            comparison_result = f"{entity1} ({value1}) vs {entity2} ({value2})"
        
        claim = f"Comparison: {comparison_result}"
        
        # Find source rows for both entities
        entity1_rows = []
        entity2_rows = []
        for i, row in enumerate(source_data):
            if any(str(entity1) in str(v) for v in row.values()):
                entity1_rows.append(i)
            if any(str(entity2) in str(v) for v in row.values()):
                entity2_rows.append(i)
        
        evidence_data = [
            {"entity": entity1, "value": value1, "source_rows": entity1_rows},
            {"entity": entity2, "value": value2, "source_rows": entity2_rows},
            {"comparison": comparison_result}
        ]
        
        return self.create_evidence(
            claim=claim,
            data=evidence_data,
            method=f"{comparison_type} comparison",
            source_rows=entity1_rows + entity2_rows
        )
    
    def track_trend_evidence(self, metric_name: str, time_series_data: List[Dict[str, Any]],
                           trend_direction: str, change_pct: float) -> str:
        """
        Track evidence for trend analysis.
        """
        claim = f"{metric_name} shows {trend_direction} trend ({change_pct:.1f}% change)"
        
        # Extract key data points
        if time_series_data:
            first_point = time_series_data[0]
            last_point = time_series_data[-1]
            peak = max(time_series_data, key=lambda x: x.get(metric_name, 0))
            trough = min(time_series_data, key=lambda x: x.get(metric_name, 0))
            
            evidence_data = [
                {"point": "start", "data": first_point},
                {"point": "end", "data": last_point},
                {"point": "peak", "data": peak},
                {"point": "trough", "data": trough}
            ]
        else:
            evidence_data = []
        
        method = f"Time series analysis over {len(time_series_data)} data points"
        
        return self.create_evidence(
            claim=claim,
            data=evidence_data,
            method=method,
            source_rows=list(range(len(time_series_data)))
        )
    
    def track_outlier_evidence(self, metric_name: str, outlier_value: Any,
                             outlier_entity: str, dataset_stats: Dict[str, Any],
                             source_row: int) -> str:
        """
        Track evidence for outlier detection.
        """
        mean = dataset_stats.get("mean", 0)
        std = dataset_stats.get("std", 1)
        z_score = (outlier_value - mean) / std if std != 0 else 0
        
        claim = f"{outlier_entity} is an outlier with {metric_name} = {outlier_value} (z-score: {z_score:.2f})"
        
        evidence_data = [
            {"outlier": {"entity": outlier_entity, "value": outlier_value, "z_score": z_score}},
            {"dataset_stats": dataset_stats},
            {"threshold": "z-score > 2 indicates outlier"}
        ]
        
        return self.create_evidence(
            claim=claim,
            data=evidence_data,
            method="Statistical outlier detection (z-score method)",
            source_rows=[source_row]
        )
    
    def generate_evidence_summary(self, evidence_ids: List[str]) -> Dict[str, Any]:
        """
        Generate a summary of evidence for a set of insights.
        """
        if not evidence_ids:
            return {"summary": "No evidence tracked", "details": []}
        
        details = []
        total_confidence = 0
        all_source_rows = set()
        
        for eid in evidence_ids:
            if eid in self.evidence_store:
                evidence = self.evidence_store[eid]
                details.append({
                    "claim": evidence.claim,
                    "method": evidence.calculation_method,
                    "confidence": evidence.confidence,
                    "data_points": len(evidence.supporting_data),
                    "source_rows": evidence.source_rows[:5]  # First 5 rows
                })
                total_confidence += evidence.confidence
                all_source_rows.update(evidence.source_rows)
        
        avg_confidence = total_confidence / len(details) if details else 0
        
        return {
            "summary": f"Based on {len(details)} pieces of evidence from {len(all_source_rows)} data points",
            "average_confidence": avg_confidence,
            "details": details
        }
    
    def format_evidence_backed_narrative(self, narrative: str, evidence_map: Dict[str, str]) -> str:
        """
        Format a narrative with inline evidence citations.
        evidence_map: {claim_text: evidence_id}
        """
        # Replace claims with cited versions
        cited_narrative = narrative
        
        for claim, evidence_id in evidence_map.items():
            if evidence_id in self.evidence_store:
                evidence = self.evidence_store[evidence_id]
                # Add inline citation
                citation = f"{claim} [Evidence: {len(evidence.source_rows)} data points, {evidence.confidence:.0%} confidence]"
                cited_narrative = cited_narrative.replace(claim, citation)
        
        # Add evidence appendix
        appendix = "\n\n📊 Evidence Details:\n"
        for i, (claim, evidence_id) in enumerate(evidence_map.items(), 1):
            if evidence_id in self.evidence_store:
                evidence = self.evidence_store[evidence_id]
                appendix += f"\n{i}. {claim}\n"
                appendix += f"   - Method: {evidence.calculation_method}\n"
                appendix += f"   - Data points: {len(evidence.supporting_data)}\n"
                appendix += f"   - Confidence: {evidence.confidence:.0%}\n"
        
        return cited_narrative + appendix
    
    def get_evidence_for_value(self, value: Any, column: str, 
                             result_data: List[Dict[str, Any]]) -> List[int]:
        """
        Find which rows in the result set contribute to a specific value.
        """
        contributing_rows = []
        
        for i, row in enumerate(result_data):
            if column in row and row[column] == value:
                contributing_rows.append(i)
        
        return contributing_rows
    
    def _generate_evidence_id(self, claim: str, data: List[Dict[str, Any]]) -> str:
        """Generate unique ID for evidence"""
        content = f"{claim}:{len(data)}:{str(data)[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _extract_supporting_data(self, data: List[Dict[str, Any]], claim: str) -> List[Dict[str, Any]]:
        """Extract only relevant data points for the claim"""
        # This is a simplified version - in practice would use NLP to identify relevant fields
        if len(data) <= 10:
            return data
        
        # For large datasets, sample representative points
        sample_size = min(10, len(data))
        step = len(data) // sample_size
        return [data[i] for i in range(0, len(data), step)][:sample_size]
    
    def _calculate_evidence_confidence(self, data: List[Dict[str, Any]], method: str) -> float:
        """Calculate confidence score based on evidence quality"""
        confidence = 0.5  # Base confidence
        
        # More data points increase confidence
        if len(data) >= 10:
            confidence += 0.2
        elif len(data) >= 5:
            confidence += 0.1
        
        # Known reliable methods increase confidence
        reliable_methods = ["direct calculation", "statistical analysis", "time series analysis"]
        if any(m in method.lower() for m in reliable_methods):
            confidence += 0.2
        
        # Check for data completeness (no nulls)
        if all(all(v is not None for v in d.values()) for d in data if isinstance(d, dict)):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _describe_aggregation_method(self, agg_type: str, value_count: int, result: Any) -> str:
        """Generate human-readable description of aggregation method"""
        descriptions = {
            "SUM": f"Added {value_count} values to get total of {result}",
            "AVG": f"Calculated average of {value_count} values = {result}",
            "COUNT": f"Counted {result} matching records",
            "MAX": f"Found maximum value of {result} from {value_count} values",
            "MIN": f"Found minimum value of {result} from {value_count} values"
        }
        
        return descriptions.get(agg_type.upper(), f"{agg_type} aggregation of {value_count} values")
    
    def export_evidence_report(self) -> Dict[str, Any]:
        """Export complete evidence report for audit trail"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "evidence_count": len(self.evidence_store),
            "link_count": len(self.evidence_links),
            "evidence_records": [
                {
                    "id": eid,
                    "claim": ev.claim,
                    "method": ev.calculation_method,
                    "confidence": ev.confidence,
                    "data_points": len(ev.supporting_data),
                    "created_at": ev.created_at.isoformat()
                }
                for eid, ev in self.evidence_store.items()
            ],
            "insight_links": [
                {
                    "insight": link.insight_text[:100] + "..." if len(link.insight_text) > 100 else link.insight_text,
                    "evidence_count": len(link.evidence_ids),
                    "confidence": link.confidence_score
                }
                for link in self.evidence_links
            ]
        }
