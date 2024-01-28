def handle_raw_datasets(raw_output : str, input: str) -> str :
    chunk_ouput = raw_output[len(input):]
    return chunk_ouput.split("#####")[0].strip()

def handle_llama2_datasets(raw_output : str, input: str) -> str :
    chunk_ouput = raw_output[len(input):]
    return chunk_ouput.split("[]")[0].strip()
