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

# aotumatically generate
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
        loading multiply JSON
    
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
                print(f"successfully load: {file_path}")
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
    
    # 检查第一个数据的类型
    first_data = json_data[0]
    
    # 情况1: 数据是字典列表 [{dict1, dict2}, {dict3, dict4}]
    if isinstance(first_data, list) and all(isinstance(item, dict) for item in first_data):
        # 验证所有数据都是字典列表
        if all(isinstance(data, list) and all(isinstance(item, dict) for item in data) for data in json_data):
            return 'list'
    
    # 情况2: 数据是包含statistics和detailed_results的评分文件
    elif isinstance(first_data, dict) and 'statistics' in first_data and 'detailed_results' in first_data:
        # 验证所有数据都包含这两个字段
        if all(isinstance(data, dict) and 'statistics' in data and 'detailed_results' in data for data in json_data):
            return 'grade'
    
    # 情况3: 数据是包含detailed_results字段的字典（但不是grade文件）
    elif isinstance(first_data, dict) and 'detailed_results' in first_data:
        # 验证所有数据都包含detailed_results字段
        if all(isinstance(data, dict) and 'detailed_results' in data for data in json_data):
            return 'detailed_results'
    
    # 情况4: 数据是包含questions字段的字典
    elif isinstance(first_data, dict) and 'questions' in first_data:
        # 验证所有数据都包含questions字段
        if all(isinstance(data, dict) and 'questions' in data for data in json_data):
            return 'questions'
    
    return 'unknown'


def fuzzy_match_model(model_name: str, required_models: List[str]) -> bool:
    """
    使用模糊匹配检查模型名称是否匹配必需的模型
    
    Args:
        model_name: 要检查的模型名称
        required_models: 必需的模型列表
        
    Returns:
        是否匹配
    """
    model_name_lower = model_name.lower().strip()
    
    for required_model in required_models:
        required_model_lower = required_model.lower().strip()
        
        # 检查各种可能的匹配情况
        if (required_model_lower in model_name_lower or 
            model_name_lower in required_model_lower or
            required_model_lower.replace('-', '') in model_name_lower.replace('-', '') or
            model_name_lower.replace('-', '') in required_model_lower.replace('-', '')):
            return True
    
    return False


def check_model_answers(questions_dict: Dict[str, Dict], required_models: List[str]) -> Tuple[bool, List[Dict], Dict[str, Dict]]:
    """
    检查每个问题是否包含所有必需的模型答案（使用模糊匹配）
    
    Args:
        questions_dict: 问题字典
        required_models: 必需的模型列表
        
    Returns:
        (是否所有问题都符合要求, 缺失信息列表, 不完整的问题字典)
    """
    missing_info = []
    incomplete_questions = {}
    all_valid = True
    
    for question, question_data in questions_dict.items():
        if 'answers' not in question_data:
            missing_info.append({
                'question': question,
                'issue': '缺少answers字段',
                'missing_models': required_models
            })
            all_valid = False
            incomplete_questions[question] = question_data.copy()
            continue
        
        existing_models = list(question_data['answers'].keys())
        
        # 检查每个必需的模型是否有匹配
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
                'issue': '缺少部分模型答案',
                'missing_models': missing_models,
                'existing_models': existing_models
            })
            all_valid = False
            incomplete_questions[question] = question_data.copy()
    
    return all_valid, missing_info, incomplete_questions


def separate_complete_incomplete_questions(questions_dict: Dict[str, Dict], required_models: List[str]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    将问题分为完整和不完整两部分（使用模糊匹配）
    
    Args:
        questions_dict: 原始问题字典
        required_models: 必需的模型列表
        
    Returns:
        (完整的问题字典, 不完整的问题字典)
    """
    complete_questions = {}
    incomplete_questions = {}
    
    for question, question_data in questions_dict.items():
        if 'answers' not in question_data:
            incomplete_questions[question] = question_data.copy()
            continue
        
        existing_models = list(question_data['answers'].keys())
        
        # 检查每个必需的模型是否有匹配
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
    合并字典列表
    
    Args:
        json_data: 字典列表的列表
        
    Returns:
        合并后的字典列表
    """
    merged_list = []
    
    for data_list in json_data:
        merged_list.extend(data_list)
    
    return merged_list


def calculate_score_distribution(detailed_results: List[Dict]) -> Dict[str, int]:
    """
    计算分数分布
    
    Args:
        detailed_results: 详细结果列表
        
    Returns:
        分数分布字典
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
            score = result['avg_scores']['total'] * 2  # 转换为100分制
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
    合并包含statistics和detailed_results的评分文件
    
    Args:
        json_data: 包含statistics和detailed_results的字典列表
        
    Returns:
        合并后的字典
    """
    if not json_data:
        return {}
    
    # 使用第一个字典作为基础
    merged_dict = json_data[0].copy()
    merged_detailed_results = []
    
    # 合并所有detailed_results
    for data in json_data:
        if 'detailed_results' in data and isinstance(data['detailed_results'], list):
            merged_detailed_results.extend(data['detailed_results'])
    
    # 更新merged_dict
    merged_dict['detailed_results'] = merged_detailed_results
    
    # 重新计算统计信息
    if 'statistics' in merged_dict:
        stats = merged_dict['statistics']
        
        # 重新计算总问题数
        stats['total_questions'] = len(merged_detailed_results)
        
        # 统计有效评分
        valid_count = 0
        total_scores = []
        
        for result in merged_detailed_results:
            if 'avg_scores' in result and 'total' in result['avg_scores']:
                valid_count += 1
                total_scores.append(result['avg_scores']['total'])
        
        stats['valid_grades'] = valid_count
        stats['failed_grades'] = stats['total_questions'] - valid_count
        
        # 重新计算平均分
        if total_scores:
            stats['total_average'] = sum(total_scores) / len(total_scores)
            stats['total_average_100'] = stats['total_average'] * 2
        
        # 重新计算分数分布
        stats['score_distribution'] = calculate_score_distribution(merged_detailed_results)
        
        # 更新完成时间
        stats['completion_time'] = datetime.now().isoformat()
        
        # 如果有field_statistics，也更新它
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
    合并包含questions字段的字典
    
    Args:
        json_data: 包含questions字段的字典列表
        
    Returns:
        合并后的字典
    """
    if not json_data:
        return {}
    
    # 使用第一个字典作为基础
    merged_dict = json_data[0].copy()
    merged_questions = merged_dict.get('questions', {}).copy()
    
    # 合并所有questions
    for data in json_data[1:]:  # 从第二个开始，因为第一个已经作为基础
        if 'questions' in data and isinstance(data['questions'], dict):
            for question_key, question_data in data['questions'].items():
                if question_key in merged_questions:
                    # 如果问题已存在，需要合并answers
                    if 'answers' in merged_questions[question_key] and 'answers' in question_data:
                        # 合并answers字典
                        merged_questions[question_key]['answers'].update(question_data['answers'])
                    # 保留其他字段（如categories等）
                    for key, value in question_data.items():
                        if key != 'answers':
                            merged_questions[question_key][key] = value
                else:
                    # 如果问题不存在，直接添加
                    merged_questions[question_key] = question_data.copy()
    
    # 更新merged_dict
    merged_dict['questions'] = merged_questions
    
    return merged_dict


def rename_default_fields(data: Union[List[Dict], Dict]) -> Union[List[Dict], Dict]:
    """
    重命名default_reply和default_prompt字段为combination_1格式
    
    Args:
        data: 需要处理的数据
        
    Returns:
        处理后的数据
    """
    if isinstance(data, list):
        return [rename_default_fields(item) for item in data]
    elif isinstance(data, dict):
        # 处理字典
        processed_dict = {}
        
        # 处理字段重命名
        for key, value in data.items():
            if key == 'default_reply':
                new_key = "combination_1_reply"
                processed_dict[new_key] = value
            elif key == 'default_prompt':
                new_key = "combination_1_prompt"
                processed_dict[new_key] = value
            else:
                # 递归处理嵌套的字典或列表
                processed_dict[key] = rename_default_fields(value)
        
        return processed_dict
    else:
        return data


def save_json(data: Union[List[Dict], Dict], output_path: str) -> None:
    """
    保存JSON数据到文件
    
    Args:
        data: 要保存的数据
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"合并结果已保存到: {output_path}")
    except Exception as e:
        print(f"错误: 保存文件时发生错误: {e}")
        sys.exit(1)


def save_incomplete_questions(incomplete_questions: Dict[str, Dict], incomplete_info: List[Dict], output_path: str) -> None:
    """
    保存不完整的问题到单独的JSON文件
    
    Args:
        incomplete_questions: 不完整的问题字典
        incomplete_info: 缺失信息列表
        output_path: 输出文件路径
    """
    # 创建包含详细信息的数据结构
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
        print(f"不完整的问题已保存到: {output_path}")
    except Exception as e:
        print(f"错误: 保存不完整问题文件时发生错误: {e}")


def main():
    # ==============================================
    # 主程序逻辑
    # ==============================================
    
    # 也可以通过命令行参数覆盖配置
    parser = argparse.ArgumentParser(description='合并多个JSON文件')
    parser.add_argument('files', nargs='*', help='要合并的JSON文件路径（可选，会覆盖脚本中的配置）')
    parser.add_argument('-o', '--output', help='输出文件路径（可选，会覆盖脚本中的配置）')
    parser.add_argument('--preview', action='store_true', help='仅预览合并结果，不保存文件')
    parser.add_argument('--rename-defaults', action='store_true', help='重命名default_reply和default_prompt字段为第一个combination格式')
    parser.add_argument('--check-models', action='store_true', help='检查每个问题是否包含所有必需的模型答案')
    parser.add_argument('--no-check-models', action='store_true', help='跳过模型答案完整性检查')
    parser.add_argument('--incomplete-output', help='不完整问题的输出文件路径')
    
    args = parser.parse_args()
    
    # 使用命令行参数覆盖默认配置（如果提供）
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
    
    # 检查文件数量
    if len(files_to_merge) < 2:
        print("错误: 请提供2-4个JSON文件")
        sys.exit(1)
    
    # 检查文件是否存在
    for file_path in files_to_merge:
        if not Path(file_path).exists():
            print(f"错误: 文件 {file_path} 不存在")
            sys.exit(1)
    
    # 检查是否为multi_model_answer类型文件
    is_multi_model_file = any('multi_model_answer' in str(f) for f in files_to_merge)
    
    # 检查是否为grade类型文件
    is_grade_file = any('grade' in str(f).lower() for f in files_to_merge)
    
    print(f"准备合并 {len(files_to_merge)} 个JSON文件:")
    for i, file_path in enumerate(files_to_merge, 1):
        print(f"  {i}. {file_path}")
    print()
    
    # 加载JSON文件
    json_data = load_json_files(files_to_merge)
    
    # 统计每个文件的问题数量
    file_stats = []
    for i, data in enumerate(json_data):
        if isinstance(data, dict) and 'questions' in data:
            question_count = len(data['questions'])
            file_stats.append((files_to_merge[i].name, question_count))
    
    # 检测合并类型
    merge_type = detect_merge_type(json_data)
    
    if merge_type == 'unknown':
        print("错误: 无法识别JSON数据格式。请确保所有文件都是相同的格式：")
        print("1. 字典列表格式: [{dict1, dict2}, {dict3, dict4}]")
        print("2. 包含detailed_results字段的字典格式")
        print("3. 包含questions字段的字典格式")
        print("4. 包含statistics和detailed_results的评分文件格式")
        sys.exit(1)
    
    # 执行合并
    if merge_type == 'list':
        print("检测到字典列表格式，正在合并...")
        merged_result = merge_dict_lists(json_data)
        print(f"合并完成，共合并了 {len(merged_result)} 个字典项")
        
    elif merge_type == 'grade':
        print("检测到评分文件格式，正在合并...")
        merged_result = merge_grade_files(json_data)
        total_items = len(merged_result.get('detailed_results', []))
        
        # 打印每个文件的统计信息
        print("\n📊 文件统计:")
        for i, data in enumerate(json_data):
            filename = files_to_merge[i].name
            question_count = len(data.get('detailed_results', []))
            avg_score = data.get('statistics', {}).get('total_average_100', 0)
            print(f"  - {filename}: {question_count} 个问题, 平均分: {avg_score:.2f}")
        
        # 打印合并后的统计
        if 'statistics' in merged_result:
            stats = merged_result['statistics']
            print(f"\n📊 合并后统计:")
            print(f"  - 总问题数: {stats['total_questions']}")
            print(f"  - 有效评分: {stats['valid_grades']}")
            print(f"  - 平均分: {stats.get('total_average_100', 0):.2f}")
            print(f"\n📊 分数分布:")
            for range_key, count in stats['score_distribution'].items():
                print(f"  - {range_key}: {count} 个")
    
    elif merge_type == 'detailed_results':
        print("检测到包含detailed_results的字典格式，正在合并...")
        merged_result = merge_detailed_results(json_data)
        total_items = len(merged_result.get('detailed_results', []))
        print(f"合并完成，detailed_results中共有 {total_items} 个项目")
        
    elif merge_type == 'questions':
        print("检测到包含questions的字典格式，正在合并...")
        merged_result = merge_questions(json_data)
        total_questions = len(merged_result.get('questions', {}))
        
        # 打印每个文件的统计信息
        print("\n📊 文件统计:")
        for filename, count in file_stats:
            print(f"  - {filename}: {count} 个问题")
        print(f"  - 合并后总计: {total_questions} 个问题")
        
        # 如果是multi_model_answer文件且启用了检查，进行模型完整性检查
        if is_multi_model_file and check_model_completeness and merge_type == 'questions':
            print("\n🔍 检查模型答案完整性...")
            print(f"必需的模型: {', '.join(required_models)}")
            
            all_valid, missing_info, incomplete_questions = check_model_answers(merged_result.get('questions', {}), required_models)
            
            if all_valid:
                print("\n✅ 所有问题都包含必需的模型答案！")
            else:
                print(f"\n⚠️  发现 {len(missing_info)} 个不完整的问题")
                
                # 简洁显示缺失的问题
                print("\n不完整问题列表:")
                for i, info in enumerate(missing_info[:10], 1):  # 只显示前10个
                    print(f"{i}. {info['question'][:60]}...")
                    print(f"   缺失: {', '.join(info['missing_models'])}")
                
                if len(missing_info) > 10:
                    print(f"\n... 还有 {len(missing_info) - 10} 个不完整的问题")
                
                # 如果需要将不完整的题目单独保存
                if save_incomplete_separately and not preview_only:
                    # 分离完整和不完整的问题
                    complete_questions, incomplete_questions = separate_complete_incomplete_questions(
                        merged_result.get('questions', {}), required_models
                    )
                    
                    # 更新merged_result，只保留完整的问题
                    merged_result['questions'] = complete_questions
                    print(f"\n将把 {len(complete_questions)} 个完整的问题保存到主文件")
                    print(f"将把 {len(incomplete_questions)} 个不完整的问题保存到单独文件")
                    
                    # 保存不完整的问题
                    save_incomplete_questions(incomplete_questions, missing_info, incomplete_output_file)
                
                if not preview_only and not save_incomplete_separately:
                    response = input("\n是否继续保存文件（包含不完整的问题）？(y/n): ")
                    if response.lower() != 'y':
                        print("已取消保存操作")
                        sys.exit(0)
    
    # 重命名default字段（如果需要）
    if rename_default_fields_flag:
        merged_result = rename_default_fields(merged_result)
    
    # 预览或保存结果
    if preview_only:
        print("\n✅ 预览模式完成")
        print_merge_summary(merged_result, merge_type, len(files_to_merge))
    else:
        save_json(merged_result, output_file)
        print("\n✅ 合并成功完成！")
        print_merge_summary(merged_result, merge_type, len(files_to_merge))


def print_merge_summary(merged_result: Union[List[Dict], Dict], merge_type: str, file_count: int):
    """
    打印合并结果的详细统计信息
    
    Args:
        merged_result: 合并后的结果
        merge_type: 合并类型
        file_count: 合并的文件数量
    """
    if merge_type == 'questions':
        questions = merged_result.get('questions', {})
        total_questions = len(questions)
        
        # 统计模型数量
        model_count = set()
        for question_data in questions.values():
            if 'answers' in question_data:
                for model_name in question_data['answers'].keys():
                    model_count.add(model_name)
        
        print(f"\n📊 最终统计:")
        print(f"  - 合并文件数: {file_count} 个")
        print(f"  - 总问题数: {total_questions} 个")
        print(f"  - 涉及模型数: {len(model_count)} 个")
    elif merge_type == 'grade':
        if 'statistics' in merged_result:
            stats = merged_result['statistics']
            print(f"\n📊 最终统计:")
            print(f"  - 合并文件数: {file_count} 个")
            print(f"  - 总问题数: {stats['total_questions']} 个")
            print(f"  - 有效评分: {stats['valid_grades']} 个")
            print(f"  - 失败评分: {stats['failed_grades']} 个")
            print(f"  - 总平均分: {stats.get('total_average_100', 0):.2f} 分")
    else:
        print(f"\n📊 最终统计:")
        print(f"  - 合并文件数: {file_count} 个")
        print(f"  - 数据格式: {merge_type}")


if __name__ == "__main__":

    main()   
