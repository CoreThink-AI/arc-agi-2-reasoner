"""
Object comparison utilities for analyzing changes between input and output grids.

This module provides functionality to compare two lists of BaseObject dictionary representations
and identify which objects were added, removed, or retained.
"""

from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass


@dataclass
class ObjectComparison:
    """Results of comparing two object lists."""
    added: List[Dict[str, Any]]
    removed: List[Dict[str, Any]]
    retained: List[Tuple[Dict[str, Any], Dict[str, Any]]]  # (before, after) pairs


def get_object_signature(obj_dict: Dict[str, Any]) -> Tuple:
    """
    Extract a signature tuple from an object dictionary for comparison.
    
    Only compares: color, placement, centroid, x1, x2, y1, y2, bottom_left, top_right
    
    Args:
        obj_dict: Dictionary representation of a BaseObject
        
    Returns:
        Tuple containing the comparison attributes
    """
    return (
        obj_dict.get('color'),
        tuple(obj_dict.get('placement', [])),
        tuple(obj_dict.get('centroid', [])),
        obj_dict.get('x1'),
        obj_dict.get('x2'),
        obj_dict.get('y1'),
        obj_dict.get('y2'),
        tuple(obj_dict.get('bottom_left', [])),
        tuple(obj_dict.get('top_right', []))
    )


def find_similar_objects(target_obj: Dict[str, Any], obj_list: List[Dict[str, Any]], tolerance: float = 0.0) -> List[int]:
    """
    Find objects in obj_list that are similar to target_obj.
    
    Args:
        target_obj: Object to find matches for
        obj_list: List of objects to search in
        tolerance: Tolerance for coordinate comparison (default: exact match)
        
    Returns:
        List of indices of similar objects
    """
    target_sig = get_object_signature(target_obj)
    similar_indices = []
    
    for i, obj in enumerate(obj_list):
        obj_sig = get_object_signature(obj)
        
        # Check if objects are similar
        if objects_similar(target_sig, obj_sig, tolerance):
            similar_indices.append(i)
    
    return similar_indices


def objects_similar(sig1: Tuple, sig2: Tuple, tolerance: float = 0.0) -> bool:
    """
    Check if two object signatures are similar within tolerance.
    
    Args:
        sig1: First object signature
        sig2: Second object signature
        tolerance: Tolerance for numeric comparisons
        
    Returns:
        True if objects are considered similar
    """
    if len(sig1) != len(sig2):
        return False
    
    for val1, val2 in zip(sig1, sig2):
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > tolerance:
                return False
        elif isinstance(val1, tuple) and isinstance(val2, tuple):
            if len(val1) != len(val2):
                return False
            for v1, v2 in zip(val1, val2):
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if abs(v1 - v2) > tolerance:
                        return False
                elif v1 != v2:
                    return False
        elif val1 != val2:
            return False
    
    return True


def compare_object_lists(before_list: List[Dict[str, Any]], 
                        after_list: List[Dict[str, Any]], 
                        tolerance: float = 0.0) -> ObjectComparison:
    """
    Compare two lists of object dictionaries and identify changes.
    
    Args:
        before_list: List of objects from input grid
        after_list: List of objects from output grid
        tolerance: Tolerance for coordinate comparison
        
    Returns:
        ObjectComparison containing added, removed, and retained objects
    """
    added = []
    removed = []
    retained = []
    
    # Track which objects have been matched
    before_matched = set()
    after_matched = set()
    
    # Find retained objects (objects that exist in both lists)
    for i, before_obj in enumerate(before_list):
        similar_indices = find_similar_objects(before_obj, after_list, tolerance)
        
        if similar_indices:
            # Object exists in both lists - it's retained
            # Use the first match (could be improved with better matching logic)
            j = similar_indices[0]
            retained.append((before_obj, after_list[j]))
            before_matched.add(i)
            after_matched.add(j)
    
    # Find removed objects (in before but not in after)
    for i, before_obj in enumerate(before_list):
        if i not in before_matched:
            removed.append(before_obj)
    
    # Find added objects (in after but not in before)
    for j, after_obj in enumerate(after_list):
        if j not in after_matched:
            added.append(after_obj)
    
    return ObjectComparison(added=added, removed=removed, retained=retained)


def analyze_changes(comparison: ObjectComparison) -> Dict[str, Any]:
    """
    Analyze the changes between object lists and provide summary statistics.
    
    Args:
        comparison: ObjectComparison result
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_added': len(comparison.added),
        'total_removed': len(comparison.removed),
        'total_retained': len(comparison.retained),
        'net_change': len(comparison.added) - len(comparison.removed),
        'changes_detected': len(comparison.added) > 0 or len(comparison.removed) > 0
    }
    
    # Analyze color changes in retained objects
    color_changes = []
    position_changes = []
    
    for before_obj, after_obj in comparison.retained:
        before_color = before_obj.get('color')
        after_color = after_obj.get('color')
        
        if before_color != after_color:
            color_changes.append({
                'from_color': before_color,
                'to_color': after_color,
                'object': after_obj
            })
        
        # Check for position changes
        before_centroid = before_obj.get('centroid', [])
        after_centroid = after_obj.get('centroid', [])
        
        if before_centroid != after_centroid:
            position_changes.append({
                'from_position': before_centroid,
                'to_position': after_centroid,
                'object': after_obj
            })
    
    analysis['color_changes'] = color_changes
    analysis['position_changes'] = position_changes
    analysis['total_color_changes'] = len(color_changes)
    analysis['total_position_changes'] = len(position_changes)
    
    return analysis


def print_comparison_summary(comparison: ObjectComparison, analysis: Dict[str, Any] = None):
    """
    Print a human-readable summary of the object comparison.
    
    Args:
        comparison: ObjectComparison result
        analysis: Optional analysis results
    """
    if analysis is None:
        analysis = analyze_changes(comparison)
    
    print("=== Object Comparison Summary ===")
    print(f"Added objects: {analysis['total_added']}")
    print(f"Removed objects: {analysis['total_removed']}")
    print(f"Retained objects: {analysis['total_retained']}")
    print(f"Net change: {analysis['net_change']}")
    print(f"Changes detected: {analysis['changes_detected']}")
    
    if analysis['total_color_changes'] > 0:
        print(f"\nColor changes: {analysis['total_color_changes']}")
        for change in analysis['color_changes']:
            print(f"  {change['from_color']} -> {change['to_color']}")
    
    if analysis['total_position_changes'] > 0:
        print(f"\nPosition changes: {analysis['total_position_changes']}")
        for change in analysis['position_changes']:
            print(f"  {change['from_position']} -> {change['to_position']}")


# Example usage and testing
if __name__ == "__main__":
    # Example object dictionaries (simplified for testing)
    before_objects = [
        {
            'color': 1,
            'placement': (0, 0),
            'centroid': (1, 1),
            'x1': 0, 'x2': 2, 'y1': 0, 'y2': 2,
            'bottom_left': (2, 0),
            'top_right': (0, 2)
        },
        {
            'color': 2,
            'placement': (3, 3),
            'centroid': (4, 4),
            'x1': 3, 'x2': 5, 'y1': 3, 'y2': 5,
            'bottom_left': (5, 3),
            'top_right': (3, 5)
        }
    ]
    
    after_objects = [
        {
            'color': 1,
            'placement': (0, 0),
            'centroid': (1, 1),
            'x1': 0, 'x2': 2, 'y1': 0, 'y2': 2,
            'bottom_left': (2, 0),
            'top_right': (0, 2)
        },
        {
            'color': 3,  # Color changed from 2 to 3
            'placement': (3, 3),
            'centroid': (4, 4),
            'x1': 3, 'x2': 5, 'y1': 3, 'y2': 5,
            'bottom_left': (5, 3),
            'top_right': (3, 5)
        },
        {
            'color': 4,  # New object
            'placement': (6, 6),
            'centroid': (7, 7),
            'x1': 6, 'x2': 8, 'y1': 6, 'y2': 8,
            'bottom_left': (8, 6),
            'top_right': (6, 8)
        }
    ]
    
    comparison = compare_object_lists(before_objects, after_objects)
    analysis = analyze_changes(comparison)
    print_comparison_summary(comparison, analysis)
