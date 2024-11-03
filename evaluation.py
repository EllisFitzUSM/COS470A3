from ranx import Qrels, Run, evaluate
import argparse as ap
import pandas as pd

def __main__():
    parser = ap.ArgumentParser()
    parser.add_argument('results_files', nargs='+')
    args = parser.parse_args()

    qrels: Qrels = Qrels.from_file('qrels/qrel_test.tsv', kind='trec')

    for index, results_file in enumerate(args.results_files):
        mean_run: Run = Run.from_file(results_file, kind='trec')
        print(results_file)
        evaluate(qrels, mean_run, ['precision@1', 'precision@5', 'ndcg@5', 'mrr', 'map'], return_mean=True)
        print(mean_run.mean_scores)
        print('\n------------------------------------------------------------------------------------------------\n')
        ski_jump_run: Run = Run.from_file(results_file, kind='trec')
        evaluate(qrels, ski_jump_run, 'precision@5', return_mean=False)
        pd.DataFrame(ski_jump_run.scores['precision@5'], columns=['qID', 'precision@5']).to_csv(F'ski_jump_run_{index+1}.csv')


if __name__ == '__main__':
    __main__()