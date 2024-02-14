import { DataArray, pipeline } from "@xenova/transformers";

let result = 0;

const runner = async () => {
  let extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
  );

  const output = await extractor("This is a simple test.", {
    pooling: "mean",
    normalize: true,
  });

  const output1 = await extractor("This is a simple test.", {
    pooling: "mean",
    normalize: true,
  });

  const similarity = await dotProduct({
    values: output.data,
    values1: output1.data,
  });
  console.log("similarity", similarity);
};

runner();

const isD1Array = <T>(value: (T | T[] | T[][])[]): value is T[] => {
  return !Array.isArray(value[0]);
};

const dotProduct = async ({
  values,
  values1,
}: {
  values: DataArray;
  values1: DataArray;
}) => {
  if (values.length != values1.length) {
    throw new Error("Both Parameters must match in length");
  }
  for (let index in values) {
    result += values[index] * values1[index];
  }
  return result;
};
