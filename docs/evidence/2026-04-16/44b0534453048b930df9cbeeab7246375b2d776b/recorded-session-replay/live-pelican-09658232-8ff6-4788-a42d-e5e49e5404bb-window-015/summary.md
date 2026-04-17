# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015`
- winner mode: `learned_route`
- trace hash: `sha256-6e0ff46b57f7c50af20d23a4d8a3f648535a36cc4021c3842ecad10617366b5b`
- fixture hash: `sha256-d2c3dec4ca105c441633ffddbfc56cdd05e4790ddeac1ae7cb9c9e93a7fe734a`
- score hash: `sha256-3d70f1808d9bee0e4974a45a61d00021c33e25937a7c353d1daf6c586f42eb42`
- bundle hash: `sha256-b1d13458478a4c58a96c75fb007f2c0762153f058e1d462ff6492510501cdb2e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-48620826d5383928480bcd6e76b64505c9f9a20a21654ee2da4ad581ffb660b0 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c667238c2fe5019e4127f1c29021b8a8c3c83183db092c8300164c85f857961 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5d97ead9083fb589e016d5612be01640fb2c446bd1991bdcf936dbdd390cfcc1 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-a7b8ed23a00d92c5f3d16bcbcb373b58fd21d3895f049e8222e16b62e1926085 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-6b82f183 | sha256-13524f6311abba3b476ebfd28955cc845f27856dba76b3f142b511feb14a5458 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6b82f183 | sha256-d1bbc01e92fd82054d1165a9204635ff46c5b0490642b69d69548505229658ab |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-c9997fe2 | sha256-99f0e598f486f11cd56f41b2f67cac2c4494486473acff63f1a102ae88bab060 |
