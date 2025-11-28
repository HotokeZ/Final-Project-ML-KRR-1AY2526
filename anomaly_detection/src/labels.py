import pandas as pd
import os


def describe_cluster(df, cluster_label, top_n=3, pct_thresh=0.5):
    sub = df[df['cluster'] == cluster_label]
    n = len(sub)
    lines = []
    lines.append(f'### Cluster {cluster_label} — {n} rows ({n/len(df):.2%} of anomalies)\n')

    # Categorical highlights: for each categorical column, pick top value and pct
    cat_cols = sub.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['Timestamp']]
    highlights = []
    for c in cat_cols:
        vc = sub[c].value_counts(normalize=True)
        if vc.empty:
            continue
        top_val = vc.index[0]
        top_pct = vc.iloc[0]
        if top_pct >= pct_thresh:
            highlights.append(f'{int(top_pct*100)}% {c} = {top_val}')
    if highlights:
        lines.append('- Strong categorical signals: ' + '; '.join(highlights) + '\n')

    # More nuanced top categories (top 2–3)
    more = []
    for c in cat_cols:
        vc = sub[c].value_counts(normalize=True).head(top_n)
        if len(vc) > 1:
            vals = ', '.join([f"{i} ({p:.1%})" for i, p in vc.items()])
            more.append(f'{c}: {vals}')
    if more:
        lines.append('- Top categorical distributions:\n')
        for m in more:
            lines.append(f'  - {m}\n')

    # Numeric summaries
    num_cols = sub.select_dtypes(include=['number']).columns.tolist()
    if num_cols:
        lines.append('- Numeric summaries (means):\n')
        for c in num_cols:
            lines.append(f'  - {c}: {sub[c].mean():.4f}\n')

    # Representative examples (first 3 rows)
    lines.append('\n- Example rows:\n')
    ex = sub.head(3).drop(columns=[c for c in ['anomaly_score', 'is_anomaly', 'cluster'] if c in sub.columns])
    for i, row in ex.iterrows():
        row_text = ', '.join([f'{col}={row[col]}' for col in ex.columns])
        lines.append(f'  - {row_text}\n')

    lines.append('\n')
    return ''.join(lines)


def generate_labels(clusters_csv, out_md, pct_thresh=0.6):
    df = pd.read_csv(clusters_csv)
    out_lines = []
    out_lines.append('# Cluster labels and short descriptions\n\n')
    # Ensure cluster col exists
    if 'cluster' not in df.columns:
        raise ValueError('clusters_csv must contain a `cluster` column')

    clusters = sorted(df['cluster'].unique())
    for c in clusters:
        out_lines.append(describe_cluster(df, c, pct_thresh=pct_thresh))

    os.makedirs(os.path.dirname(out_md) or '.', exist_ok=True)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
    print(f'Wrote cluster labels to {out_md}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--clusters_csv', required=True)
    p.add_argument('--out_md', default='output/cluster_labels.md')
    p.add_argument('--pct_thresh', type=float, default=0.6, help='Threshold for strong categorical signal (0-1)')
    args = p.parse_args()

    generate_labels(args.clusters_csv, args.out_md, pct_thresh=args.pct_thresh)
