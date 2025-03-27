from typing import Dict, List, Tuple, Any
import re



def parse_data_series(content: str) -> Dict[str, List[Any]]:
    data = {}
    pairs = re.findall(r'<(\w+)>(.*?)</\w+>', content)
    for key, value in pairs:
        if key not in data:
            data[key] = []
        try:
            data[key].append(float(value))
        except ValueError:
            data[key].append(value)
    return data


def parse_text_display(content: str) -> List[Dict[str, Any]]:
    elements = []
    polygons = re.findall(r'<polygon>(.*?)</polygon>', content)
    texts = re.findall(r'<text>(.*?)</text>', content)
    for polygon, text in zip(polygons, texts):
        element = {'text': text}
        coords = re.findall(r'<(\w+)>(\d+)</\w+>', polygon)
        for key, value in coords:
            element[key] = int(value)
        elements.append(element)
    return elements


def extract_conetnt_from_sequence(content: str, bos: str, eos: str) -> str:
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
            content = extract_conetnt_from_sequence(pred_str, start_tag, end_tag)
            
            if token == 'data_series':
                result[token] = parse_data_series(content)
            elif token == 'text_display':
                result[token] = parse_text_display(content)
            else:
                nested_tags = detect_nested_tags(content, token_map)
                if nested_tags:
                    result[token] = build_nested_dict(content, token_map, nested_tags)
                else:
                    try:
                        result[token] = int(content.strip())
                    except ValueError:
                        result[token] = content.strip()
    return result




def post_processing(
        pred_str: str, 
        token_map: Dict[str, List[str]], 
        token_order: List[str] = ['chart_type', 'plot_bb', 'data_series', 'text_display']) -> Dict[str, Any]:
    return build_nested_dict(pred_str, token_map, token_order)
