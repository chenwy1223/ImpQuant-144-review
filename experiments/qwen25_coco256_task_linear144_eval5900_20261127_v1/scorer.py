"""Task scoring for the offline likelihood-free accuracy comparison."""
from metrics import score_response, textvqa_score
from mmmu_local import eval_multi_choice, eval_open, parse_multi_choice_response, parse_open_response


def grade(row, response):
    task = row["task"]
    if task=="vizwiz_vqa_val":
        value,_=textvqa_score(response,row["target"])
        return float(value)
    if task=="mmmu_val":
        if row["answer_kind"]=="choice":
            opts=row["metadata"]["options"]
            parsed=parse_multi_choice_response(response,list(opts),opts)
            return float(eval_multi_choice(row["target"],parsed))
        return float(eval_open(row["target"],parse_open_response(response)))
    return float(score_response(task,response,row["target"],row["metadata"])["score"])
