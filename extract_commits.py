#!/usr/bin/env python3
"""
Extract per-commit stats from git repos into a CSV.

Columns: timestamp, author, lines_added, lines_removed, repo
"""

import subprocess, sys, os, csv

def _git_log(repo, since):
# Use %aI = author date, strict ISO 8601 with timezone (RFC3339)
    fmt = "%H%x09%an%x09%aI"
    cmd = ["git", "-C", repo, "log", f"--since={since}", "--numstat", f"--format={fmt}", "--no-color"]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[warn] git failed in {repo}: {e.output}\n")
        return ""

def _parse_numstat(stream, repo_name):
    cur_author=None; cur_date=None; add_sum=0; del_sum=0; in_commit=False
    for line in stream.splitlines():
        if not line.strip(): continue
        parts=line.split("\t")
        if len(parts)==3 and ":" in parts[2]:  # header
            if in_commit:
                yield {"timestamp":cur_date,"author":cur_author,
                       "lines_added":add_sum,"lines_removed":del_sum,"repo":repo_name}
            _, cur_author, cur_date = parts
            add_sum=0; del_sum=0; in_commit=True
        elif len(parts)>=3:
            a,d=parts[0],parts[1]
            try:
                a=0 if a=="-" else int(a)
                d=0 if d=="-" else int(d)
            except ValueError:
                a,d=0,0
            add_sum+=a; del_sum+=d
    if in_commit:
        yield {"timestamp":cur_date,"author":cur_author,
               "lines_added":add_sum,"lines_removed":del_sum,"repo":repo_name}

def discover_repos(paths):
    repos=[]
    for p in paths:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, ".git")):
            repos.append(p)
        elif os.path.isdir(p):
            for name in os.listdir(p):
                d=os.path.join(p,name)
                if os.path.isdir(os.path.join(d,".git")): repos.append(d)
    return sorted(set(repos))

def main():
    if len(sys.argv)<2:
        sys.exit("Usage: extract_commits.py <repo_or_parent_dir> [...]")
    repos=discover_repos(sys.argv[1:])
    rows=[]
    for repo in repos:
        name=os.path.basename(os.path.abspath(repo))
        raw=_git_log(repo,"2 years ago")
        if raw: rows.extend(_parse_numstat(raw,name))
    if not rows: sys.exit("No commits found.")
    with open("commits.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["timestamp","author","lines_added","lines_removed","repo"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote commits.csv with {len(rows)} rows from {len(repos)} repos.")

if __name__=="__main__":
    main()

