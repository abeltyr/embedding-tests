import { HfInference } from "@huggingface/inference";

const hf = new HfInference(process.env.HF_TOKEN);
console.log(process.env.HF_TOKEN);

let result = 0;

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

const runner = async () => {
  const output = await hf.featureExtraction({
    model: "avsolatorio/GIST-Embedding-v0",
    inputs: "That is a happy person over here",
  });

  console.log("output", output);
  //   const output1 = await hf.featureExtraction({
  //     model: "avsolatorio/GIST-Embedding-v0",
  //     inputs: "That is a happy person over here",
  //   });

  //   console.log("output1", output1);

  //   if (isD1Array(output) && isD1Array(output1)) {
  //     const finalRanking = dotProduct({ values: output, values1: output1 });
  //     console.log("finalRanking", finalRanking);
  //   }
  //   const [result] = output;

  //   if (Array.isArray(result)) {
  //     console.log("output", result.length);
  //   }

  //   const output2 = await hf.featureExtraction({
  //     model: "sentence-transformers/all-MiniLM-L6-v2",
  //     inputs: "That is a happy person over here",
  //   });

  //   console.log("output2", output2);
  //   const [result1] = output2;

  //   if (Array.isArray(result1)) {
  //     console.log("output2", result1.length);
  //   }
};

runner();

const isD1Array = <T>(value: (T | T[] | T[][])[]): value is T[] => {
  return !Array.isArray(value[0]);
};
