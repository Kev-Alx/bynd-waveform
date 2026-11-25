"""Historical native source scoring, scoped to one registered expanded phase."""
import argparse
import source_evaluate_refine as engine
import scaled_scope as scope

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--name',required=True)
    args,_=p.parse_known_args()
    engine.cycle=scope.evaluator_scope(args.name);engine.RUN48=scope.RUN
    engine.main()
