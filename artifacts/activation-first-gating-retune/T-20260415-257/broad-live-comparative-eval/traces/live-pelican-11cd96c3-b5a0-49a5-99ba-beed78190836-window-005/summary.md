# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e98818ee1fbfe6af19470aba80e3474e972af078ccda49d0b283bf9b3f9cdf05`
- fixture hash: `sha256-d657d23463bf41cd4159e478f5223c1f2880e97d1b0706959b1e80d3f0d4e745`
- score hash: `sha256-a64e33218216e1fdf40bb52db47b5af00a6619bcedf9a035e23702484a85f585`
- bundle hash: `sha256-8168eff9775784d876fb968f10b9623ea16a9f17fe92d13da3388da01a65a9b9`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eaf48763997381e6d9ac148445f4fc78050abde4363c03acd4f6f65040d7cf98 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df36b28b988c74c24c22b1db104f6fa3327cb938c787cbb570e9c1d9ed599fc2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a895cd863f232f897f6f193a316379d8daeb5aa7966d012707d5e7a168d48936 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b55be26f97f5b1b49e01e0267fa51799e566f3ae997088c8eee2a683cc510294 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8e254d53 | sha256-84db317e70f9df70fa3bf7ffc2fd3cd059070bb1caac331e941dbbc39fd0a76e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8e254d53 | sha256-02172432b7701643c6a4a33b756544a3fad6b70901bc1bce16b7450f24b158aa |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5999e2d0 | sha256-2022247047c66c324f653f51cce2af7ec98b21ef2aa0141bd5f5d458011de5b0 |
