import { HfInference } from "@huggingface/inference";
import { config } from "dotenv";

config({ path: ".env.local" });

const hf = new HfInference(process.env.HF_TOKEN);
console.log(process.env.HF_TOKEN);

let result = 0;

const runner = async () => {
  try {
    const output = await hf.featureExtraction({
      model: "BAAI/bge-base-en-v1.5",
      inputs: "The cat is running around chasing the red light",
    });

    const output1 = await hf.featureExtraction({
      model: "BAAI/bge-base-en-v1.5",
      inputs: "The dog is running around",
    });

    console.log("output1", output1);

    if (isD1Array(output) && isD1Array(output1)) {
      const finalRanking = await dotProduct({
        values: output,
        values1: output1,
      });
      console.log("finalRanking", finalRanking);
    }
  } catch (e) {
    console.log("error", e);
  }
};

runner();

const isD1Array = <T>(value: (T | T[] | T[][])[]): value is T[] => {
  return !Array.isArray(value[0]);
};

const dotProduct = async ({
  values,
  values1,
}: {
  values: number[];
  values1: number[];
}) => {
  if (values.length != values1.length) {
    throw new Error("Both Parameters must match in length");
  }
  for (let index in values) {
    result += values[index] * values1[index];
  }
  return result;
};
