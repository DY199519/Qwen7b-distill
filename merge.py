import json
import argparse
import sys
from typing import List, Dict, Any, Union, Set, Tuple
from pathlib import Path
from datetime import datetime

# ==============================================
# 📁 modify the folder path
# ==============================================

# basic directory configuration
BASE_DIR = Path(r"D:\project7\prompt")
BASE_DIR_1 = Path(r"D:\project7")
BASE_DIR_2 = Path(r"D:\project7\merge10000")

# merge the JSON files
files_to_merge = [
    BASE_DIR_2 / "grades-3+1-1-3600.json",      
    BASE_DIR_2 / "grades-3+1-3600-4400.json",       
    BASE_DIR_2 / "grades-3+1-4400-5000.json",       
    BASE_DIR_2 / "grades-3+1-5000-5800.json",      
    BASE_DIR_2 / "grades-3+1-5800-6300.json",      
    BASE_DIR_2 / "grades-3+1-6300-6800.json",    
    BASE_DIR_2 / "grades-3+1-6800-7800.json",       
    BASE_DIR_2 / "grades-3+1-7800-8100.json",       
    BASE_DIR_2 / "grades-3+1-8100-8600.json",       
    BASE_DIR_2 / "grades-3+1-8600-8900.json",      
    BASE_DIR_2 / "grades-3+1-8900-9400.json",       

]
# output file path
output_file = BASE_DIR_2 / "grades-3+1-1-9400.json"

# automatically generate
incomplete_output_file = BASE_DIR_2 / "incomplete_questions1-3600.json"

# ==============================================
# ⚙️ modify function options
# ==============================================

preview_only = False

rename_default_fields_flag = True

# Whether to check the integrity of the model answers (only valid for files of the multi_model_answer type)
check_model_completeness = True

# Required model list (using fuzzy matching)
required_models = ["doubao-pro", "gemini-2.5-flash", "grok-3"]

# Whether to save incomplete questions separately
save_incomplete_separately = True


def load_json_files(file_paths: List[str]) -> List[Dict]:
    """
        loading multiple JSON files
    
    Args:
        file_paths: 
        
    Returns:
        List of loaded JSON data
    """
    json_data = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_data.append(data)
                print(f"successfully loaded: {file_path}")
        except FileNotFoundError:
            print(f"ERROR: The file {file_path} does not exist.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: The file {file_path} is not valid JSON: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: An error occurred while reading the file {file_path}: {e}")
            sys.exit(1)
    
    return json_data


def detect_merge_type(json_data: List[Dict]) -> str:
    """
    Detect merge type
    
    Args:
        json_data: 
        
    Returns:
        'list' 
        'detailed_results' 
        'questions' 
        'grade'
    """
    if not json_data:
        return 'unknown'
    
    # Check the type of the first data entry
    first_data = json_data[0]
    
    # Case 1: Data is a list of dictionaries [{dict1, dict2}, {dict3, dict4}]
    if isinstance(first_data, list) and all(isinstance(item, dict) for item in first_data):
        # Verify all data entries are lists of dictionaries
        if all(isinstance(data, list) and all(isinstance(item, dict) for item in data) for data in json_data):
            return 'list'
    
    # Case 2: Data is a grading file containing statistics and detailed_results
    elif isinstance(first_data, dict) and 'statistics' in first_data and 'detailed_results' in first_data:
        # Verify all data entries contain these two fields
        if all(isinstance(data, dict) and 'statistics' in data and 'detailed_results' in data for data in json_data):
            return 'grade'
    
    # Case 3: Data is a dictionary containing the detailed_results field (but not a grade file)
    elif isinstance(first_data, dict) and 'detailed_results' in first_data:
        # Verify all data entries contain the detailed_results field
        if all(isinstance(data, dict) and 'detailed_results' in data for data in json_data):
            return 'detailed_results'
    
    # Case 4: Data is a dictionary containing the questions field
    elif isinstance(first_data, dict) and 'questions' in first_data:
        # Verify all data entries contain the questions field
        if all(isinstance(data, dict) and 'questions' in data for data in json_data):
            return 'questions'
    
    return 'unknown'


def fuzzy_match_model(model_name: str, required_models: List[str]) -> bool:
    """
    Check if model name matches required models using fuzzy matching
    
    Args:
        model_name: Model name to check
        required_models: List of required models
        
    Returns:
        Whether there is a match
    """
    model_name_lower = model_name.lower().strip()
    
    for required_model in required_models:
        required_model_lower = required_model.lower().strip()
        
        # Check various possible matching scenarios
        if (required_model_lower in model_name_lower or 
            model_name_lower in required_model_lower or
            required_model_lower.replace('-', '') in model_name_lower.replace('-', '') or
            model_name_lower.replace('-', '') in required_model_lower.replace('-', '')):
            return True
    
    return False


def check_model_answers(questions_dict: Dict[str, Dict], required_models: List[str]) -> Tuple[bool, List[Dict], Dict[str, Dict]]:
    """
    Check if each question contains answers from all required models (using fuzzy matching)
    
    Args:
        questions_dict: Question dictionary
        required_models: List of required models
        
    Returns:
        (Whether all questions meet requirements, list of missing information, dictionary of incomplete questions)
    """
    missing_info = []
    incomplete_questions = {}
    all_valid = True
    
    for question, question_data in questions_dict.items():
        if 'answers' not in question_data:
            missing_info.append({
                'question': question,
                'issue': 'Missing answers field',
                'missing_models': required_models
            })
            all_valid = False
            incomplete_questions[question] = question_data.copy()
            continue
        
        existing_models = list(question_data['answers'].keys())
        
        # Check if each required model has a match
        missing_models = []
        for required_model in required_models:
            found = False
            for existing_model in existing_models:
                if fuzzy_match_model(existing_model, [required_model]):
                    found = True
                    break
            if not found:
                missing_models.append(required_model)
        
        if missing_models:
            missing_info.append({
                'question': question,
                'issue': 'Missing answers from some models',
                'missing_models': missing_models,
                'existing_models': existing_models
            })
            all_valid = False
            incomplete_questions[question] = question_data.copy()
    
    return all_valid, missing_info, incomplete_questions


def separate_complete_incomplete_questions(questions_dict: Dict[str, Dict], required_models: List[str]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Separate questions into complete and incomplete (using fuzzy matching)
    
    Args:
        questions_dict: Original question dictionary
        required_models: List of required models
        
    Returns:
        (Dictionary of complete questions, Dictionary of incomplete questions)
    """
    complete_questions = {}
    incomplete_questions = {}
    
    for question, question_data in questions_dict.items():
        if 'answers' not in question_data:
            incomplete_questions[question] = question_data.copy()
            continue
        
        existing_models = list(question_data['answers'].keys())
        
        # Check if each required model has a match
        missing_models = []
        for required_model in required_models:
            found = False
            for existing_model in existing_models:
                if fuzzy_match_model(existing_model, [required_model]):
                    found = True
                    break
            if not found:
                missing_models.append(required_model)
        
        if missing_models:
            incomplete_questions[question] = question_data.copy()
        else:
            complete_questions[question] = question_data.copy()
    
    return complete_questions, incomplete_questions


def merge_dict_lists(json_data: List[List[Dict]]) -> List[Dict]:
    """
    Merge lists of dictionaries
    
    Args:
        json_data: List of lists of dictionaries
        
    Returns:
        Merged list of dictionaries
    """
    merged_list = []
    
    for data_list in json_data:
        merged_list.extend(data_list)
    
    return merged_list


def calculate_score_distribution(detailed_results: List[Dict]) -> Dict[str, int]:
    """
    Calculate score distribution
    
    Args:
        detailed_results: List of detailed results
        
    Returns:
        Score distribution dictionary
    """
    distribution = {
        "0-20": 0,
        "20-30": 0,
        "30-40": 0,
        "40-50": 0
    }
    
    for result in detailed_results:
        if 'avg_score_100' in result:
            score = result['avg_score_100']
        elif 'avg_scores' in result and 'total' in result['avg_scores']:
            score = result['avg_scores']['total'] * 2  # Convert to 100-point scale
        else:
            continue
            
        if score < 20:
            distribution["0-20"] += 1
        elif score < 30:
            distribution["20-30"] += 1
        elif score < 40:
            distribution["30-40"] += 1
        elif score <= 50:
            distribution["40-50"] += 1
    
    return distribution


def merge_grade_files(json_data: List[Dict]) -> Dict:
    """
    Merge grade files containing statistics and detailed_results
    
    Args:
        json_data: List of dictionaries containing statistics and detailed_results
        
    Returns:
        Merged dictionary
    """
    if not json_data:
        return {}
    
    # Use the first dictionary as the base
    merged_dict = json_data[0].copy()
    merged_detailed_results = []
    
    # Merge all detailed_results
    for data in json_data:
        if 'detailed_results' in data and isinstance(data['detailed_results'], list):
            merged_detailed_results.extend(data['detailed_results'])
    
    # Update merged_dict
    merged_dict['detailed_results'] = merged_detailed_results
    
    # Recalculate statistics
    if 'statistics' in merged_dict:
        stats = merged_dict['statistics']
        
        # Recalculate total number of questions
        stats['total_questions'] = len(merged_detailed_results)
        
        # Count valid grades
        valid_count = 0
        total_scores = []
        
        for result in merged_detailed_results:
            if 'avg_scores' in result and 'total' in result['avg_scores']:
                valid_count += 1
                total_scores.append(result['avg_scores']['total'])
        
        stats['valid_grades'] = valid_count
        stats['failed_grades'] = stats['total_questions'] - valid_count
        
        # Recalculate average score
        if total_scores:
            stats['total_average'] = sum(total_scores) / len(total_scores)
            stats['total_average_100'] = stats['total_average'] * 2
        
        # Recalculate score distribution
        stats['score_distribution'] = calculate_score_distribution(merged_detailed_results)
        
        # Update completion time
        stats['completion_time'] = datetime.now().isoformat()
        
        # Update field_statistics if present
        if 'field_statistics' in stats:
            for field_name in stats['field_statistics']:
                stats['field_statistics'][field_name]['count'] = valid_count
                if total_scores:
                    stats['field_statistics'][field_name]['average'] = stats['total_average']
                    stats['field_statistics'][field_name]['average_100'] = stats['total_average_100']
    
    return merged_dict


#def merge_detailed_results(json_data: List[Dict]) -> Dict:


def merge_questions(json_data: List[Dict]) -> Dict:
    """
    Merge dictionaries containing the questions field
    
    Args:
        json_data: List of dictionaries containing the questions field
        
    Returns:
        Merged dictionary
    """
    if not json_data:
        return {}
    
    # Use the first dictionary as the base
    merged_dict = json_data[0].copy()
    merged_questions = merged_dict.get('questions', {}).copy()
    
    # Merge all questions
    for data in json_data[1:]:  # Start from the second one since the first is used as base
        if 'questions' in data and isinstance(data['questions'], dict):
            for question_key, question_data in data['questions'].items():
                if question_key in merged_questions:
                    # If question exists, need to merge answers
                    if 'answers' in merged_questions[question_key] and 'answers' in question_data:
                        # Merge answers dictionary
                        merged_questions[question_key]['answers'].update(question_data['answers'])
                    # Preserve other fields (like categories, etc.)
                    for key, value in question_data.items():
                        if key != 'answers':
                            merged_questions[question_key][key] = value
                else:
                    # If question doesn't exist, add it directly
                    merged_questions[question_key] = question_data.copy()
    
    # Update merged_dict
    merged_dict['questions'] = merged_questions
    
    return merged_dict


def rename_default_fields(data: Union[List[Dict], Dict]) -> Union[List[Dict], Dict]:
    """
    Rename default_reply and default_prompt fields to combination_1 format
    
    Args:
        data: Data to process
        
    Returns:
        Processed data
    """
    if isinstance(data, list):
        return [rename_default_fields(item) for item in data]
    elif isinstance(data, dict):
        # Process dictionary
        processed_dict = {}
        
        # Handle field renaming
        for key, value in data.items():
            if key == 'default_reply':
                new_key = "combination_1_reply"
                processed_dict[new_key] = value
            elif key == 'default_prompt':
                new_key = "combination_1_prompt"
                processed_dict[new_key] = value
            else:
                # Recursively process nested dictionaries or lists
                processed_dict[key] = rename_default_fields(value)
        
        return processed_dict
    else:
        return data


def save_json(data: Union[List[Dict], Dict], output_path: str) -> None:
    """
    Save JSON data to file
    
    Args:
        data: Data to save
        output_path: Output file path
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Merge results saved to: {output_path}")
    except Exception as e:
        print(f"Error: An error occurred while saving the file: {e}")
        sys.exit(1)


def save_incomplete_questions(incomplete_questions: Dict[str, Dict], incomplete_info: List[Dict], output_path: str) -> None:
    """
    Save incomplete questions to a separate JSON file
    
    Args:
        incomplete_questions: Dictionary of incomplete questions
        incomplete_info: List of missing information
        output_path: Output file path
    """
    # Create data structure containing detailed information
    incomplete_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_incomplete_questions": len(incomplete_questions),
            "required_models": required_models,
            "source_files": [str(f) for f in files_to_merge]
        },
        "questions": incomplete_questions,
        "missing_info": incomplete_info
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(incomplete_data, f, ensure_ascii=False, indent=2)
        print(f"Incomplete questions saved to: {output_path}")
    except Exception as e:
        print(f"Error: An error occurred while saving the incomplete questions file: {e}")


def main():
    # ==============================================
    # Main program logic
    # ==============================================
    
    # Can also override configuration via command line arguments
    parser = argparse.ArgumentParser(description='Merge multiple JSON files')
    parser.add_argument('files', nargs='*', help='Paths to JSON files to merge (optional, overrides script configuration)')
    parser.add_argument('-o', '--output', help='Output file path (optional, overrides script configuration)')
    parser.add_argument('--preview', action='store_true', help='Only preview merge results, do not save file')
    parser.add_argument('--rename-defaults', action='store_true', help='Rename default_reply and default_prompt fields to first combination format')
    parser.add_argument('--check-models', action='store_true', help='Check if each question contains answers from all required models')
    parser.add_argument('--no-check-models', action='store_true', help='Skip model answer completeness check')
    parser.add_argument('--incomplete-output', help='Output file path for incomplete questions')
    
    args = parser.parse_args()
    
    # Override default configuration with command line arguments if provided
    if args.files:
        global files_to_merge
        files_to_merge = args.files
    if args.output:
        global output_file
        output_file = args.output
    if args.preview:
        global preview_only
        preview_only = True
    if args.rename_defaults:
        global rename_default_fields_flag
        rename_default_fields_flag = True
    if args.check_models:
        global check_model_completeness
        check_model_completeness = True
    if args.no_check_models:
        check_model_completeness = False
    if args.incomplete_output:
        global incomplete_output_file
        incomplete_output_file = args.incomplete_output
    
    # Check number of files
    if len(files_to_merge) < 2:
        print("Error: Please provide 2-4 JSON files")
        sys.exit(1)
    
    # Check if files exist
    for file_path in files_to_merge:
        if not Path(file_path).exists():
            print(f"Error: File {file_path} does not exist")
            sys.exit(1)
    
    # Check if files are of multi_model_answer type
    is_multi_model_file = any('multi_model_answer' in str(f) for f in files_to_merge)
    
    # Check if files are of grade type
    is_grade_file = any('grade' in str(f).lower() for f in files_to_merge)
    
    print(f"Preparing to merge {len(files_to_merge)} JSON files:")
    for i, file_path in enumerate(files_to_merge, 1):
        print(f"  {i}. {file_path}")
    print()
    
    # Load JSON files
    json_data = load_json_files(files_to_merge)
    
    # Count number of questions in each file
    file_stats = []
    for i, data in enumerate(json_data):
        if isinstance(data, dict) and 'questions' in data:
            question_count = len(data['questions'])
            file_stats.append((files_to_merge[i].name, question_count))
    
    # Detect merge type
    merge_type = detect_merge_type(json_data)
    
    if merge_type == 'unknown':
        print("Error: Unable to recognize JSON data format. Please ensure all files are in the same format:")
        print("1. List of dictionaries format: [{dict1, dict2}, {dict3, dict4}]")
        print("2. Dictionary format containing detailed_results field")
        print("3. Dictionary format containing questions field")
        print("4. Grade file format containing statistics and detailed_results")
        sys.exit(1)
    
    # Perform merge
    if merge_type == 'list':
        print("Detected list of dictionaries format, merging...")
        merged_result = merge_dict_lists(json_data)
        print(f"Merge completed, merged a total of {len(merged_result)} dictionary items")
        
    elif merge_type == 'grade':
        print("Detected grade file format, merging...")
        merged_result = merge_grade_files(json_data)
        total_items = len(merged_result.get('detailed_results', []))
        
        # Print statistics for each file
        print("\n📊 File statistics:")
        for i, data in enumerate(json_data):
            filename = files_to_merge[i].name
            question_count = len(data.get('detailed_results', []))
            avg_score = data.get('statistics', {}).get('total_average_100', 0)
            print(f"  - {filename}: {question_count} questions, average score: {avg_score:.2f}")
        
        # Print merged statistics
        if 'statistics' in merged_result:
            stats = merged_result['statistics']
            print(f"\n📊 Merged statistics:")
            print(f"  - Total questions: {stats['total_questions']}")
            print(f"  - Valid grades: {stats['valid_grades']}")
            print(f"  - Average score: {stats.get('total_average_100', 0):.2f}")
            print(f"\n📊 Score distribution:")
            for range_key, count in stats['score_distribution'].items():
                print(f"  - {range_key}: {count} items")
    
    elif merge_type == 'detailed_results':
        print("Detected dictionary format containing detailed_results, merging...")
        merged_result = merge_detailed_results(json_data)
        total_items = len(merged_result.get('detailed_results', []))
        print(f"Merge completed, detailed_results contains {total_items} items")
        
    elif merge_type == 'questions':
        print("Detected dictionary format containing questions, merging...")
        merged_result = merge_questions(json_data)
        total_questions = len(merged_result.get('questions', {}))
        
        # Print statistics for each file
        print("\n📊 File statistics:")
        for filename, count in file_stats:
            print(f"  - {filename}: {count} questions")
        print(f"  - Total after merging: {total_questions} questions")
        
        # If it's a multi_model_answer file and checking is enabled, perform model completeness check
        if is_multi_model_file and check_model_completeness and merge_type == 'questions':
            print("\n🔍 Checking model answer completeness...")
            print(f"Required models: {', '.join(required_models)}")
            
            all_valid, missing_info, incomplete_questions = check_model_answers(merged_result.get('questions', {}), required_models)
            
            if all_valid:
                print("\n✅ All questions contain answers from required models!")
            else:
                print(f"\n⚠️  Found {len(missing_info)} incomplete questions")
                
                # Briefly display missing questions
                print("\nIncomplete question list:")
                for i, info in enumerate(missing_info[:10], 1):  # Only show first 10
                    print(f"{i}. {info['question'][:60]}...")
                    print(f"   Missing: {', '.join(info['missing_models'])}")
                
                if len(missing_info) > 10:
                    print(f"\n... and {len(missing_info) - 10} more incomplete questions")
                
                # If need to save incomplete questions separately
                if save_incomplete_separately and not preview_only:
                    # Separate complete and incomplete questions
                    complete_questions, incomplete_questions = separate_complete_incomplete_questions(
                        merged_result.get('questions', {}), required_models
                    )
                    
                    # Update merged_result to only keep complete questions
                    merged_result['questions'] = complete_questions
                    print(f"\nWill save {len(complete_questions)} complete questions to main file")
                    print(f"Will save {len(incomplete_questions)} incomplete questions to separate file")
                    
                    # Save incomplete questions
                    save_incomplete_questions(incomplete_questions, missing_info, incomplete_output_file)
                
                if not preview_only and not save_incomplete_separately:
                    response = input("\nDo you want to continue saving the file (including incomplete questions)? (y/n): ")
                    if response.lower() != 'y':
                        print("Save operation cancelled")
                        sys.exit(0)
    
    # Rename default fields if needed
    if rename_default_fields_flag:
        merged_result = rename_default_fields(merged_result)
    
    # Preview or save results
    if preview_only:
        print("\n✅ Preview mode completed")
        print_merge_summary(merged_result, merge_type, len(files_to_merge))
    else:
        save_json(merged_result, output_file)
        print("\n✅ Merge completed successfully!")
        print_merge_summary(merged_result, merge_type, len(files_to_merge))


def print_merge_summary(merged_result: Union[List[Dict], Dict], merge_type: str, file_count: int):
    """
    Print detailed statistics of merge results
    
    Args:
        merged_result: Merged result
        merge_type: Type of merge
        file_count: Number of merged files
    """
    if merge_type == 'questions':
        questions = merged_result.get('questions', {})
        total_questions = len(questions)
        
        # Count number of models
        model_count = set()
        for question_data in questions.values():
            if 'answers' in question_data:
                for model_name in question_data['answers'].keys():
                    model_count.add(model_name)
        
        print(f"\n📊 Final statistics:")
        print(f"  - Number of merged files: {file_count}")
        print(f"  - Total questions: {total_questions}")
        print(f"  - Number of models involved: {len(model_count)}")
    elif merge_type == 'grade':
        if 'statistics' in merged_result:
            stats = merged_result['statistics']
            print(f"\n📊 Final statistics:")
            print(f"  - Number of merged files: {file_count}")
            print(f"  - Total questions: {stats['total_questions']}")
            print(f"  - Valid grades: {stats['valid_grades']}")
            print(f"  - Failed grades: {stats['failed_grades']}")
            print(f"  - Overall average score: {stats.get('total_average_100', 0):.2f} points")
    else:
        print(f"\n📊 Final statistics:")
        print(f"  - Number of merged files: {file_count}")
        print(f"  - Data format: {merge_type}")


if __name__ == "__main__":

    main()   
