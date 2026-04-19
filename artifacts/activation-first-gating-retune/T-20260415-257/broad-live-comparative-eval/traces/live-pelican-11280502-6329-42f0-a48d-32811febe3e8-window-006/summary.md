# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d39e73987cbab37d8621de930c333f3c7648788fff43ee43b47bd6d1ef64fb83`
- fixture hash: `sha256-a074cddd6e9248ce93aa0306421d446e0244240e7cd7f2087a7d75eb50352127`
- score hash: `sha256-fd0556bd5f58a96239646a1aa6461d67f27bf53aa1cb05f483924018527e3997`
- bundle hash: `sha256-ab79ebbbe07eb13b6214971f7cb254174ea874de61aa164d7bf45c15ad32f01f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2f53febf0867c1eaf1f16249136eed66aac774d903a17b5eedd4459dd80af44d |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-4c45b9ce6b57bbf7b72e5fc1ff6da60670e9b3d0d447ae1b632a778cfdc1806d |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-fc2a04ab56f7ee0b066f2f90d8a7c32d03b9ef884850b0aef2b7f5e6ca730162 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-62c748886210498155b7d91b04d5b0e1c0632009edcb9454f6ea2ef80ae72772 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-876d487b | sha256-8503ea8e931cbe1cbb171b942ffcd31195d67f655772b4e0c9b9e5476619e1f5 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-876d487b | sha256-06fe276f7ba9af6174508373116782cd7dd99e0128d4881045bd8826e8ef3962 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-876d487b | sha256-8503ea8e931cbe1cbb171b942ffcd31195d67f655772b4e0c9b9e5476619e1f5 |
