# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-53e7f7c2a908bfa01e8a36f987e9389c06b6f1c4270256cec14da19431b1dd8e`
- fixture hash: `sha256-4dd26bce21297c56105a43961b6bacbe27d7812f2b72d27dc4b8b7698e0474b9`
- score hash: `sha256-23aebce269716bef2052becde55fe530f03f759f85e9689ef7569c115d2f26f0`
- bundle hash: `sha256-142781cea318acd6906a264f971a70c0b06bcaba97c54134c3fe19111dcf557e`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aeccda1f8aefa0b00a23d8464e4e2bbf0fb55e8c49bf77bf016cce252f0ffad2 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2f51aaf0766bfb18117c15bcdefe0081bc9403aa24c68ac2a5671a04a58000a0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-732748298af1db6054efa4793907eb0fdd1a97127cc6be5c847e7045e85840b6 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-9b47089eedc35de8986594e52870e856b3f7d6722402654022eb75a2447b83eb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bd139731 | sha256-9465bd23c39a1f5ebe98fda769fa6e6a67bf2e5b1cd9619d3dc1d18f1313fe62 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bd139731 | sha256-744df5043a5a9a1e96d6da70c09163bdd973c34f6ac69d636a5486ee81da7d99 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-1e32ea86 | sha256-e42d2eb8c1e8cea460b3584363b4cee1d3d06936f72bab3d7c5cfb6ab414fdc8 |
