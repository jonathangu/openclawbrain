# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f8bd6e98ba409d4b92ff33d315c90158dc9f7928f49ee95918b29862594fc07`
- fixture hash: `sha256-f2d0f492e33718dcda5e95309dd8b8ae83d2a012ce623b86565c773255e59638`
- score hash: `sha256-fd4fc6dbf95b351b91cba7bdbaf1317619638528cfa36fae1bf48d2a2d113976`
- bundle hash: `sha256-a368ee0e7535199aa70f377bc7b55058ba4d61ec35a550bb3ad9e4a0936fcf31`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3a49035e9fd3e0717342039595aabed753c46d3f982a6fbdc847832f0114d10f |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-44ab04f5cc3f1d6e5b58e4e94ed84ffb134b9b2881d624e8018570daf41453bf |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bd72b398734aa233d2049af717733195afbb3f381ec726528c8e46450aa86473 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-1f9a8f686dd94efcb2bc67a33a3e4fe15301ee982db7341aebdb24782314ac82 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b4f09955 | sha256-503235937fd4ee8b39ae637b8964554cc6297d0384b0a7b84281c08906410c3f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b4f09955 | sha256-e8e6688835adf4e9169ba1239c186f0e89bdf9e6ab15ca9e964058e68d6c49c7 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-604c14e6 | sha256-08eb53ddcd78680c47c72b4c804fe4e393dcc7e6ea165194d70984944378d860 |
