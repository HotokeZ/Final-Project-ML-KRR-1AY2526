import pandas as pd
import os


def summarize(df, out_path):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    # remove technical or score columns
    exclude = set(['Timestamp', 'anomaly_score', 'is_anomaly'])
    cat_cols = [c for c in cat_cols if c not in exclude]
    num_cols = [c for c in num_cols if c not in exclude]

    n = len(df)
    an = df[df['is_anomaly'] == True]
    na = df[df['is_anomaly'] == False]

    lines = []
    lines.append('# Anomaly Summary\n\n')
    lines.append(f'- Total rows: {n}\n')
    lines.append(f'- Anomalies: {len(an)} ({len(an)/n:.2%})\n\n')

    lines.append('## Categorical columns\n\n')
    for c in cat_cols:
        lines.append(f'### {c}\n\n')
        lines.append('| value | anomalies % | normal % | diff |\n')
        lines.append('|---|---:|---:|---:|\n')
        vc_an = an[c].value_counts(normalize=True)
        vc_na = na[c].value_counts(normalize=True)
        vals = list(dict.fromkeys(list(vc_an.index) + list(vc_na.index)))
        for v in vals:
            a_pct = vc_an.get(v, 0.0)
            n_pct = vc_na.get(v, 0.0)
            lines.append(f'| {v} | {a_pct:.3%} | {n_pct:.3%} | {a_pct-n_pct:+.3%} |\n')
        lines.append('\n')

    if num_cols:
        lines.append('## Numeric columns\n\n')
        for c in num_cols:
            a_mean = an[c].mean()
            n_mean = na[c].mean()
            lines.append(f'- **{c}** — anomalies mean: {a_mean:.4f}, normal mean: {n_mean:.4f}, diff: {a_mean-n_mean:.4f}\n')
        lines.append('\n')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'Wrote report to {out_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scores_csv', required=True)
    p.add_argument('--out_md', default='output/anomaly_report.md')
    args = p.parse_args()

    df = pd.read_csv(args.scores_csv)
    # ensure boolean
    if 'is_anomaly' in df.columns and df['is_anomaly'].dtype != bool:
        try:
            df['is_anomaly'] = df['is_anomaly'].astype(bool)
        except Exception:
            df['is_anomaly'] = df['is_anomaly'].map({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False})
    summarize(df, args.out_md)
