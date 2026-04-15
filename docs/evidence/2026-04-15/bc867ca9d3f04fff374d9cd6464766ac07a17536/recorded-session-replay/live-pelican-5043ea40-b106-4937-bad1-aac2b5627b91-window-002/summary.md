# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-347a801443da9e3d23f8dc976f3d286dbcc3cafa0984aebf1f93ff8efbfd1773`
- fixture hash: `sha256-3e9f54e7049625692dd39972563612e44cc8adf4a2a27dc80d450c5621a5caf7`
- score hash: `sha256-a9a14e892ed5638c24332115e028b56052019f634d123ca62280ce64a26b9a16`
- bundle hash: `sha256-d28f5da4e30c2e7b04c2db5d91b3baa49fdf37c1f563fd5c3da5a382a4c35fc6`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bc058e6f191036e6bf4f3884982c6a502fc3d927441bbbd1c5d745ba4e254aee |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-82efe34d7c2610c457da6490c612ab6f3be7b8bf4697a44f0d5eeba21b22b997 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f8018985c29075b7e4a6fe50403054cca82c6c233f6b4945693d925740af0212 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-a0247d94b79bc33c35923f84711b784520782d22a0f8139c7812e0e7ec5a96cf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-569d9ed8 | sha256-ec7ca60b095b0057bc98107b08df8f2e511a9bd30f6a44f9d42e3d5b36177abc |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-569d9ed8 | sha256-ec7ca60b095b0057bc98107b08df8f2e511a9bd30f6a44f9d42e3d5b36177abc |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-569d9ed8 | sha256-ec7ca60b095b0057bc98107b08df8f2e511a9bd30f6a44f9d42e3d5b36177abc |
