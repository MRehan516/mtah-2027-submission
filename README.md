# Neural Nerd works: More than a Hack 2027, Phase 1

Sticky-decoder possession labelling from the provided tactical tracking data.
The repository contains no match data.

## Reproduce
1. Install Python 3.10+, then: pip install -r requirements.txt
2. Put the provided files in data/tactical/ :
   BAR-ATH_tactical_atd.txt and BAR-ATH_tactical_atd.xml
3. Run: python mtah_code/run_v2.py --team Neural Nerd works [FLAGS USED]
4. Output: submission/Neural Nerd works.csv ; check with evaluation/validate_submission.py from the organisers' repository.

Files in mtah_code: features.py, pipeline.py, improve.py, run_v2.py, sanity.py, synthetic_bench.py.
Code in starter/ and games.py is from the organisers' MIT-licensed repository.
