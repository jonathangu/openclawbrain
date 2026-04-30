import { roundUtility } from "./rubric.ts";
export type Uncertainty = { n: number; mean: number; standardError: number | null; lowNWarning: boolean };
export function describeUncertainty(values: number[]): Uncertainty {
  if (!values.length) return { n: 0, mean: 0, standardError: null, lowNWarning: true };
  const mean = roundUtility(values.reduce((s,v)=>s+v,0)/values.length);
  const variance = values.length > 1 ? values.reduce((s,v)=>s+(v-mean)**2,0)/(values.length-1) : 0;
  return { n: values.length, mean, standardError: values.length > 1 ? roundUtility(Math.sqrt(variance/values.length)) : null, lowNWarning: values.length < 6 };
}
