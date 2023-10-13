import traceback
from typing import List, Dict, Any


def set_hits_json(text_list: List[str], y_hat_label_list: List[List[str]], y_hat_prob_list: List[List[float]]) -> Dict:
    def set_document_predict_top_k(document_label: List[str], document_prob: List[float]) -> Dict:
        document_top_k_json = {'inference': []}
        key = 'predict'
        for idx in range(len(document_label)):
            value = {
                'rank': idx + 1,
                'category': document_label[idx],
                'probability': document_prob[idx],
            }
            document_top_k_json['inference'].append({key: value})

        return document_top_k_json

    hits = {'documents': []}
    for idx in range(len(y_hat_label_list)):
        document_label = y_hat_label_list[idx]
        document_prob = y_hat_prob_list[idx]

        hits['documents'].append({
            'index': idx,
            'text': text_list[idx]
        })
        inference_result_json = set_document_predict_top_k(document_label, document_prob)
        hits['documents'][idx].update({
            'inference': inference_result_json.get('inference')
        })

    return hits


def set_response_json(hits_json: Dict, predict_config: Dict[str, Any], exec_time: float) -> Dict:
    try:
        response_json = {
            "took": 1,
            "exec_time": exec_time,
            "hits": {
                "total": {
                    "value": predict_config['top_k'],
                },
                "model": predict_config['model_type'],
                "collection": predict_config['text_dataset'],
                "hits": hits_json
            }
        }

        return response_json
    except Exception:
        print(traceback.format_exc())
