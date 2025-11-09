import cupy as cp
import torch

inference_kernel = cp.RawKernel(
r'''
extern "C" __global__
void inference(float* prob, int* ans, float* aa_masses, float* dp, float* dpMass ,int* lock, int* aa_indexes, float premass, int length, float grid_size, int ncandidates, int top_k, float tol2) {
    const int AA_num = 28;
    const float tol = tol2;
    const int word_num = length - 1;
    int w = blockIdx.x;
    int dim = blockDim.x;
    int h = threadIdx.x;
    if(w == 0 || h == 0) return; // skip the first row and first column

    float maxMass = aa_masses[AA_num - 8] * h;
    int maxW = int(maxMass / grid_size);
    if(w > maxW){
        lock[w * dim + h] = 1;
        return;
    }
    if(h == 1){
        for(int i = 0; i < ncandidates; i++){
            int aa_index = aa_indexes[(h - 1) * ncandidates + i];
            float aa_mass = aa_masses[aa_index];
            if(w == int(aa_mass / grid_size)){
                int min_index = 0;
                float min_value = dp[w * dim * top_k + h * top_k];
                for (int j = 1 ; j < top_k; j++) {
                    if (dp[w * dim * top_k + h * top_k + j] < min_value) {
                        min_index = j;
                        min_value = dp[w * dim * top_k + h * top_k + j];
                    }
                }
                if (dp[w * dim * top_k + h * top_k + min_index] < prob[(h-1) * ncandidates + i]) {
                    dp[w * dim * top_k + h * top_k + min_index] = prob[(h-1) * ncandidates + i];
                    dpMass[w * dim * top_k + h * top_k + min_index] = aa_mass;
                    ans[w * dim * top_k * word_num + h * top_k * word_num + min_index * word_num] = aa_index;
                }
            }
        }
        lock[w * dim + h] = 1;             
        return;
    }

    for(int i = 0; i < ncandidates; i ++){
        int aa_index = aa_indexes[(h - 1) * ncandidates + i];
        float aa_mass = aa_masses[aa_index];
        int minw = int((w * grid_size - aa_masses[aa_index]) / grid_size);
        if(minw >= 0 && minw <= int(w * grid_size)){
            while(!atomicCAS(lock + minw * dim + h - 1, 1, 1));
            __threadfence();
            for (int k = 0 ; k < top_k; k++) {
                int preid = ans[minw * dim * top_k * word_num + (h - 1) * top_k * word_num + k * word_num + (h - 2)];
                if (preid == 0 || preid > 28) continue;
                float temp = dpMass[minw * dim * top_k + (h - 1) * top_k + k] + aa_mass;
                if ((temp >= w * grid_size && temp < (w + 1) * grid_size && h != length-1) || (h == length - 1 && temp >= premass - tol && temp <= premass + tol)){
                    float tempProb = dp[minw * dim * top_k + (h - 1) * top_k + k] + prob[(h-1) * ncandidates + i];
                    int min_index = 0;
                    float min_value = dp[w * dim * top_k + h * top_k];
                    for (int j = 1 ; j < top_k; j++) {
                        if (dp[w * dim * top_k + h * top_k + j] < min_value) {
                            min_index = j;
                            min_value = dp[w * dim * top_k + h * top_k + j];
                        }
                    }
                    if (tempProb > dp[w * dim * top_k + h * top_k + min_index]) {
                        dp[w * dim * top_k + h * top_k + min_index] = tempProb;
                        dpMass[w * dim * top_k + h * top_k + min_index] = temp;
                        for (int l = 0; l < h - 1; l++) {
                            ans[w * dim * top_k * word_num + h * top_k * word_num + min_index * word_num + l] = ans[minw * dim * top_k * word_num + (h - 1) * top_k * word_num + k * word_num + l];
                        }
                        ans[w * dim * top_k * word_num + h * top_k * word_num + min_index * word_num + (h - 1)] = aa_index;
                    }
                }
            }
        }
        minw = minw + 1;
        if(minw >= 0 && minw <= int(w * grid_size)){
            while(!atomicCAS(lock + minw * dim + h - 1, 1, 1));
            __threadfence();
            for (int k = 0 ; k < top_k; k++) {
                int preid = ans[minw * dim * top_k * word_num + (h - 1) * top_k * word_num + k * word_num + (h - 2)];
                if (preid == 0 || preid > 28) continue;
                float temp = dpMass[minw * dim * top_k + (h - 1) * top_k + k] + aa_mass;
                if ((temp >= w * grid_size && temp < (w + 1) * grid_size && h != length-1) || (h == length - 1 && temp >= premass - tol && temp <= premass + tol)){
                    float tempProb = dp[minw * dim * top_k + (h - 1) * top_k + k] + prob[(h-1) * ncandidates + i];
                    int min_index = 0;
                    float min_value = dp[w * dim * top_k + h * top_k];
                    for (int j = 1 ; j < top_k; j++) {
                        if (dp[w * dim * top_k + h * top_k + j] < min_value) {
                            min_index = j;
                            min_value = dp[w * dim * top_k + h * top_k + j];
                        }
                    }
                    if (tempProb > dp[w * dim * top_k + h * top_k + min_index]) {
                        dp[w * dim * top_k + h * top_k + min_index] = tempProb;
                        dpMass[w * dim * top_k + h * top_k + min_index] = temp;
                        for (int l = 0; l < h - 1; l++) {
                            ans[w * dim * top_k * word_num + h * top_k * word_num + min_index * word_num + l] = ans[minw * dim * top_k * word_num + (h - 1) * top_k * word_num + k * word_num + l];
                        }
                        ans[w * dim * top_k * word_num + h * top_k * word_num + min_index * word_num + (h - 1)] = aa_index;
                    }
                }
            }
        }
    }
    lock[w * dim + h] = 1;  
    return;
}
''', 'inference')


def knapsack_decode(prob: torch.Tensor, aa_indexes: torch.Tensor, aa_mass: torch.Tensor, premass: float, grid_size: float, tol: float, topk=1):
    """
    Dynamic programming decoding using knapsack algorithm on GPU with CuPy.
    Args:
        prob (torch.Tensor): Tensor of shape (word_num, K) representing the probabilities of amino acids at each position.
        aa_indexes (torch.Tensor): Tensor of shape (K,) representing the indexes of amino acids.
        aa_mass (torch.Tensor): Tensor of shape (K,) representing the masses of amino acids.
        premass (float): The precursor mass.
        grid_size (float): The size of each grid cell.
        tol (float): The tolerance for mass matching.
        topk (int): The number of top sequences to return.
        device: The device to run the computation on.
    Returns:
        dp (torch.Tensor): The dynamic programming table.
        dp_mass (torch.Tensor): The mass table corresponding to the dp table.
        pre_pathes_index (torch.Tensor): The indexes of previous amino acids for each position.
    """
    s_torch = torch.cuda.current_stream()
    s_cupy = cp.cuda.ExternalStream(s_torch.cuda_stream)
    with torch.cuda.stream(s_torch), cp.cuda.Stream(s_cupy):
        prob_cp = cp.asarray(prob) # shape (word_num, K)
        aa_indexes_cp = cp.asarray(aa_indexes).astype(cp.int32) # shape (word_num, K)
        aa_mass_cp = cp.asarray(aa_mass)
        word_num, K = prob.shape
        length = word_num + 1
        cell_num = int(premass / grid_size) + 1
        if cell_num < 1:
            return None, None, None

        dp = cp.full((cell_num, length, topk), float("-inf"), dtype=cp.float32)
        dp_mass = cp.zeros((cell_num, length, topk), dtype=cp.float32)
        pathes_indexes = cp.zeros((cell_num, length, topk, word_num), dtype=cp.int32)
        lock = cp.zeros((cell_num, length), dtype=cp.int32)

        # data initialization
        dp[0, 0, 0] = 0.0
        dp_mass[0, 0, 0] = 0.0
        lock[0, :] = 1

        block_size = cell_num
        nthreads_per_block = length
        inference_kernel((block_size,), (nthreads_per_block,),
                        (prob_cp, pathes_indexes, aa_mass_cp, dp, dp_mass, lock, aa_indexes_cp, cp.float32(premass.item()), length, cp.float32(grid_size), K, topk, cp.float32(tol)))
    torch.cuda.current_stream().wait_stream(s_torch)
    return torch.as_tensor(dp), torch.as_tensor(dp_mass), torch.as_tensor(pathes_indexes)
