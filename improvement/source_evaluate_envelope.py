"""Reuse the frozen native evaluator with the new frozen cohort/variant scope."""
import source_evaluate_refine as engine
import envelope_experiment as cycle

engine.cycle = cycle
engine.RUN48 = cycle.RUN

if __name__ == '__main__':
    engine.main()
