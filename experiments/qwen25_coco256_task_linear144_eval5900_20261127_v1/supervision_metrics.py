"""Frozen local choice/OCR rules and TextVQA EvalAI consensus; zero judge calls."""
from common import require
from frozen_vqa_eval_metric import EvalAIAnswerProcessor

def textvqa_score(response,answers):
 norm=EvalAIAnswerProcessor();pred=norm(response);a=[norm(x) for x in answers];require(len(a)==10,'TextVQA original ten annotations')
 values=[min(1.,sum(x==pred for j,x in enumerate(a) if j!=i)/3.) for i in range(10)]
 return sum(values)/10.,pred

def score_response(task,response,target,metadata):
 require(isinstance(response,str),'Literal generated response')
 if task in ['scienceqa_img2017','scienceqa_img','mmbench_en_dev']:
  p=response.lower().strip();y=target.lower().strip();parsed=p if p==y else p[0] if len(p)>=2 and p[0].isalpha() and p[1]=='.' else None;v=int(parsed==y)
  return dict(correct=v,score=float(v),parsed_choice=parsed,external_judge_calls=0,random_fallback_calls=0)
 if task=='textvqa_val':
  v,p=textvqa_score(response,metadata['aliases']);return dict(score=v,normalized_prediction=p,external_judge_calls=0,random_fallback_calls=0)
 require(task in ['ocrbench','ocrbench1000'],'Fixed local OCR task');p=response.lower().strip();answers=target if isinstance(target,list) else [target];require(answers and all(isinstance(a,str) and a.strip() for a in answers),'Nonempty original aliases')
 if metadata['dataset']=='HME100k':v=int(any(a.strip().replace('\n',' ').replace(' ','') in p.replace('\n',' ').replace(' ','') for a in answers))
 else:v=int(any(a.strip().lower().replace('\n',' ') in p.replace('\n',' ') for a in answers))
 return dict(correct=v,score=float(v),external_judge_calls=0,random_fallback_calls=0)

def generation_flags(task,response,ids,kwargs,correct,valid_letters):
 require(ids and isinstance(correct,(int,float)) and 0<=correct<=1,'Nonempty scored generation');p=response.strip().upper();parsed=p if p in valid_letters else p[0] if len(p)>=2 and p[0] in valid_letters and p[1]=='.' else None;eos=kwargs['eos_token_id']
 return dict(score=float(correct),empty_response=not bool(response.strip()),first_token_eos=ids[0]==eos,ended_with_eos=ids[-1]==eos,non_eos_generation_cap_hit=len(ids)==kwargs['max_new_tokens'] and ids[-1]!=eos,generated_tokens=len(ids),parsed_choice=parsed if valid_letters else None,choice_format_unrecognized=(parsed is None) if valid_letters else None)
