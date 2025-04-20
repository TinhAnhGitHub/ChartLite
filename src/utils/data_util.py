from typing import Dict, List, Tuple, Any
import re



def parse_data_series(content: str) -> Dict[str, List[Any]]:
    return_data = []
    pairs = re.findall(r'<(\w+)>(.*?)</\w+>', content)
    data_series = [
        [pairs[i], pairs[i+1], pairs[i+2]] for i in range(0, len(pairs), 3)
    ]
    for item_series  in data_series:
        x_item = item_series[0]
        y_item = item_series[1]
        value_item = item_series[2]

        dict_ = {
            x_item[0]: x_item[1],
            y_item[0]: y_item[1],
            value_item[0]: value_item[1]
        }
        return_data.append(dict_)
    return return_data
        


def parse_text_display(content: str) -> List[Dict[str, Any]]:
    print(f"{content=}")
    elements = []
    polygons = re.findall(r'<polygon>(.*?)</polygon>', content)
    texts = re.findall(r'<text>(.*?)</text>', content)
    roles = re.findall(r'<role>(.*?)</role>',content)
    colors = re.findall(r'<colors>(.*?)</colors>', content)
    elements = []
    for polygon, text, role, color in zip(polygons, texts, roles, colors):
        text_chart = text.split(',') if role in ['x_axis','y_axis'] else text
        print(f"{text_chart=}")
        colors_chart = color.split(',') if ',' in color else color
        element = {
            'text': text_chart,
            'role': role,
            'colors': colors_chart,
            'polygon': {}
        }
        coords = re.findall(r'<(\w+)>(\d+)</\w+>', polygon)
        for key, value in coords:
            element['polygon'][key] = value
        elements.append(element)
    return elements
        


def extract_content_from_sequence(content: str, bos: str, eos: str) -> str:
    content = content.split(bos)[1]
    content = content.split(eos)[0]
    return content


def detect_nested_tags(content: str, token_map: Dict[str, List[str]]) -> List[str]:
    nested_tags = []
    for token, tags in token_map.items():
        start_tag = tags[0].replace('<', r'\<').replace('>', r'\>')
        if re.search(start_tag, content):
            nested_tags.append(token)
    return nested_tags



def build_nested_dict(
    pred_str: str,
    token_map: Dict[str, List[str]],
    token_order: List[str]
)-> Dict[str, Any]:
    result = {}
    
    for token in token_order:
        start_tag, end_tag = token_map[token]
        if start_tag in pred_str and end_tag in pred_str:
            content = extract_content_from_sequence(pred_str, start_tag, end_tag)
            
            if token == 'data_series':
                result[token] = parse_data_series(content)
            elif token == 'text_display':
                result[token] = parse_text_display(content)
            else:
                nested_tags = detect_nested_tags(content, token_map)
                if nested_tags:
                    result[token] = build_nested_dict(content, token_map, nested_tags)
                else:
                    result[token] = content.strip()
    return result




def post_processing(
        pred_str: str, 
        token_map: Dict[str, List[str]], 
        token_order: List[str] = ['chart_type', 'plot_bb', 'data_series', 'text_display']) -> Dict[str, Any]:
    return build_nested_dict(pred_str, token_map, token_order)
