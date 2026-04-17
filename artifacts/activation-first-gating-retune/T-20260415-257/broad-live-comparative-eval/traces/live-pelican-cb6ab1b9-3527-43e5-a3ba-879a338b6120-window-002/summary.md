# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4ecb7ad01ebf51dd3aeb7754e784e70eea9f4067a9392ad81778aff88de83b03`
- fixture hash: `sha256-ad7aadbe694390cc07af980435b05bd2086d5294c79bda5f4f75ff348a4a3b75`
- score hash: `sha256-cfa4d6742726896117d9856ea9eeeb1a3455da809053484368f78d3614403f53`
- bundle hash: `sha256-b9a9d5205b2f982a5a25dc490df648a1b29780279644b58582c09bd74f2bdae9`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8a8a0e6d7dd1a7545143681fd0202299acfcab2ad5ce85ed5e5cddd516c7f67 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1e9b85dc9e0337bf45b4d4f14fa66e4f235743dbe3404ac3545682b7fc201d63 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-99306c825eea8b2266b1d1900fda76be07491f89cb78d55548d78195084e0156 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-23655b5710a647b502a5e61ce238a1acf71c8c78c7f24eab3890dc1cd6159d23 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3f8aaed0 | sha256-7803e5783f89c34c4b635f4f350e9c1c81d557b85e21e183f587487277c3a44a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3f8aaed0 | sha256-1f56513a4bf677eaa9f8eb24ff76110984490a309d9b68d189aec4792fb9b439 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-df7cce03 | sha256-83ef7bad4bfc4ba5a34f5047d278a22cefe7c36ee326b028c20088421bb2db31 |
