/**
* Interface for matrix operations
* Properties:
    * - type: Type of operation (e.g., "matmul")
    * - lhs: Left-hand side matrix with shape and dtype
    * - rhs: Right-hand side matrix with shape and dtype
    * - output: Output specification with dtype
*/
class Matrix {
    constructor(type, lhs, rhs, output) {
        this.type = type;
        this.lhs = lhs;
        this.rhs = rhs;
        this.output = output;
        this.id = Math.random().toString(36).substr(2, 9);
    }

    // Calculate total FLOPs for this matrix operation
    calculateFLOPs() {
        if (this.type !== "matmul") {
            throw new Error("Only matrix multiplication is supported currently");
        }

        const B = this.lhs.shape[0];
        const D = this.lhs.shape[1];
        const F = this.rhs.shape[1];

        return 2 * B * D * F;
    }

    // Calculate total bytes for this matrix operation
    calculateBytes() {
        const bytesPerElement = {
            "bf16": 2,
            "f32": 4,
            "int8": 1,
            "int4": 0.5,
            "fp8": 1
        };

        const lhsBytes = this.lhs.shape.reduce((acc, dim) => acc * dim, 1) *
            bytesPerElement[this.lhs.dtype];

        const rhsBytes = this.rhs.shape.reduce((acc, dim) => acc * dim, 1) *
            bytesPerElement[this.rhs.dtype];

        const B = this.lhs.shape[0];
        const F = this.rhs.shape[1];
        const outputBytes = B * F * bytesPerElement[this.output.dtype];

        return {
            lhsBytes,
            rhsBytes,
            outputBytes,
            totalBytes: lhsBytes + rhsBytes + outputBytes
        };
    }

    // Calculate arithmetic intensity
    calculateArithmeticIntensity() {
        const bytes = this.calculateBytes();
        const flops = this.calculateFLOPs();

        return flops / bytes.totalBytes;
    }
}

class TPUv5e {
    constructor() {

        this.name = "TPU v5e";
        this.type = "TPU";

        // Compute capabilities
        this.flopsPerSecondBF16 = 1.97e14;
        this.flopsPerSecondINT8 = 3.94e14;

        // Memory specifications
        this.hbmCapacity = 16e9;
        this.hbmBandwidth = 8.1e11;
        this.vmemCapacity = 128e6;
        this.vmemBandwidth = 1.78e13;

        // Interconnect specifications
        this.iciOnewayBandwidth = 4.5e10;
        this.iciBidiBandwidth = 9e10;
        this.pcieBandwidth = 1.5e10;
        this.dcnBandwidth = 2.5e10;

        // Architecture details
        this.mxuDimensions = [128, 128];
        this.coresPerChip = 1;
        this.iciTopology = "2D";
        this.maxPodSize = [16, 16];
        this.iciHopLatency = 1e-6;

        // Storage for matrices and models
        this.matrices = [];
        this.models = [];
    }

    addMatrix(type, lhs, rhs, output) {
        const matrix = new Matrix(type, lhs, rhs, output);
        this.matrices.push(matrix);
        return matrix.id;
    }

    // Calculate performance metrics for a matrix
    performanceMetrics(matrixId, multiNode = false, shardingConfig = null) {
        const matrix = this.matrices.find(m => m.id === matrixId);
        if (!matrix) {
            throw new Error(`Matrix with ID ${matrixId} not found`);
        }

        if (multiNode && !shardingConfig) {
            throw new Error("Sharding configuration required for multi-node analysis");
        }

        // Single node performance
        const singleNodeMetrics = this._calculateSingleNodeMetrics(matrix);

        // Multi-node performance if requested
        let multiNodeMetrics = null;
        if (multiNode) {
            multiNodeMetrics = this._calculateMultiNodeMetrics(matrix, shardingConfig);
        }

        return {
            singleNode: singleNodeMetrics,
            multiNode: multiNodeMetrics
        };
    }

    _calculateSingleNodeMetrics(matrix) {
        const B = matrix.lhs.shape[0];
        const D = matrix.lhs.shape[1];
        const F = matrix.rhs.shape[1];

        // Calculate bytes and FLOPs
        const bytes = matrix.calculateBytes();
        const totalFLOPs = matrix.calculateFLOPs();

        // Calculate arithmetic intensity
        const arithmeticIntensity = matrix.calculateArithmeticIntensity();

        // Calculate peak hardware intensity
        const flopsPerSecond = matrix.lhs.dtype === "int8" ?
            this.flopsPerSecondINT8 :
            this.flopsPerSecondBF16;

        const peakHardwareIntensity = flopsPerSecond / this.hbmBandwidth;

        // Determine if compute or memory bound
        const isComputeBound = arithmeticIntensity > peakHardwareIntensity;

        // Calculate time estimates
        const computeTime = totalFLOPs / flopsPerSecond;
        const memoryTime = bytes.totalBytes / this.hbmBandwidth;

        // Calculate MXU utilization
        const mxuUtilization = {
            lhsUtilization: (B % this.mxuDimensions[0] === 0) ?
                1.0 :
                (B / Math.ceil(B / this.mxuDimensions[0]) / this.mxuDimensions[0]),

            rhsUtilizationRows: (D % this.mxuDimensions[0] === 0) ?
                1.0 :
                (D / Math.ceil(D / this.mxuDimensions[0]) / this.mxuDimensions[0]),

            rhsUtilizationCols: (F % this.mxuDimensions[1] === 0) ?
                1.0 :
                (F / Math.ceil(F / this.mxuDimensions[1]) / this.mxuDimensions[1])
        };

        // Calculate VMEM metrics
        const fitsInVMEM = (bytes.lhsBytes + bytes.rhsBytes) <= this.vmemCapacity;
        const vmemComputeTime = computeTime;
        const vmemMemoryTime = bytes.totalBytes / this.vmemBandwidth;
        const vmemTotalTime = Math.max(vmemComputeTime, vmemMemoryTime);

        const vmemMetrics = {
            fitsInVMEM,
            vmemComputeTime,
            vmemMemoryTime,
            vmemTotalTime,
            vmemSpeedup: memoryTime / vmemMemoryTime
        };

        // Calculate total time estimates
        const lowerBoundTime = Math.max(computeTime, memoryTime);
        const upperBoundTime = computeTime + memoryTime;

        return {
            // Basic metrics
            totalFLOPs,
            bytes,
            arithmeticIntensity,
            peakHardwareIntensity,
            isComputeBound,

            // Time estimates
            computeTime,
            memoryTime,
            lowerBoundTime,
            upperBoundTime,

            // Hardware utilization
            mxuUtilization,

            // VMEM metrics
            vmemMetrics
        };
    }

    _calculateMultiNodeMetrics(matrix, shardingConfig) {

        const shardedMatrix = new Matrix(
            matrix.type,
            JSON.parse(JSON.stringify(matrix.lhs)),
            JSON.parse(JSON.stringify(matrix.rhs)),
            JSON.parse(JSON.stringify(matrix.output))
        );

        // Apply sharding to the matrix
        if (shardingConfig.lhsShardDim !== null) {
            shardedMatrix.lhs.shape[shardingConfig.lhsShardDim] =
                Math.ceil(matrix.lhs.shape[shardingConfig.lhsShardDim] / shardingConfig.numDevices);
        }

        if (shardingConfig.rhsShardDim !== null) {
            shardedMatrix.rhs.shape[shardingConfig.rhsShardDim] =
                Math.ceil(matrix.rhs.shape[shardingConfig.rhsShardDim] / shardingConfig.numDevices);
        }

        // Calculate per-device performance
        const perDeviceMetrics = this._calculateSingleNodeMetrics(shardedMatrix);

        // Calculate communication costs
        let communicationCost = 0;

        // For example, if we shard on D dimension, we need to do an all-reduce on the output
        if (shardingConfig.lhsShardDim === 1 || shardingConfig.rhsShardDim === 0) {
            const B = matrix.lhs.shape[0];
            const F = matrix.rhs.shape[1];
            const outputBytes = B * F *
                (matrix.output.dtype === "bf16" ? 2 :
                    matrix.output.dtype === "f32" ? 4 : 1);

            // Simple model: log2(numDevices) steps, each step communicates half the data
            const steps = Math.log2(shardingConfig.numDevices);
            communicationCost = (outputBytes / 2) * steps / this.iciBidiBandwidth;
        }

        return {
            perDeviceMetrics,
            communicationCost,
            totalTime: Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
            speedupOverSingleDevice: this._calculateSingleNodeMetrics(matrix).lowerBoundTime /
                Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
            shardingEfficiency: perDeviceMetrics.lowerBoundTime /
                Math.max(perDeviceMetrics.lowerBoundTime, communicationCost)
        };
    }

    // Generate recommendations and roofline analysis
    analyze(matrixId) {
        const matrix = this.matrices.find(m => m.id === matrixId);
        if (!matrix) {
            throw new Error(`Matrix with ID ${matrixId} not found`);
        }

        const metrics = this._calculateSingleNodeMetrics(matrix);
        const recommendations = this._generateRecommendations(matrix, metrics);
        const rooflineData = this._generateRooflineData(matrix, metrics);

        return {
            metrics,
            recommendations,
            rooflineData
        };
    }

    // Generate recommendations for improving performance
    _generateRecommendations(matrix, metrics) {
        const recommendations = [];

        // Check if we're memory bound
        if (!metrics.isComputeBound) {
            recommendations.push("Operation is memory-bound. Consider increasing batch size to improve arithmetic intensity.");

            // Recommend quantization if using bf16
            if (matrix.lhs.dtype === "bf16" && matrix.rhs.dtype === "bf16") {
                recommendations.push("Consider using int8 quantization for weights to reduce memory bandwidth requirements.");
            }

            // Recommend VMEM if available
            if (metrics.vmemMetrics.fitsInVMEM) {
                recommendations.push("Consider using VMEM to store weights for higher bandwidth access.");
            }
        }

        // Check MXU utilization
        if (metrics.mxuUtilization.lhsUtilization < 0.9 ||
            metrics.mxuUtilization.rhsUtilizationRows < 0.9 ||
            metrics.mxuUtilization.rhsUtilizationCols < 0.9) {
            recommendations.push(`Consider padding matrix dimensions to multiples of MXU dimensions (${this.mxuDimensions[0]}x${this.mxuDimensions[1]}) to improve hardware utilization.`);
        }

        // Check batch size for compute-bound operation
        const B = matrix.lhs.shape[0];
        if (B < 240 && matrix.lhs.dtype === "bf16") {
            recommendations.push("For bf16 matmul on TPU, batch size should be at least 240 to be compute-bound.");
        }

        return recommendations;
    }

    _generateRooflineData(matrix, metrics) {
        // Generate points for roofline plot
        const intensityPoints = [];
        const peakFlops = matrix.lhs.dtype === "int8" ?
            this.flopsPerSecondINT8 :
            this.flopsPerSecondBF16;

        // Generate logarithmically spaced points for x-axis (arithmetic intensity)
        for (let i = -1; i <= 4; i += 0.1) {
            const intensity = Math.pow(10, i);
            const achievableFlops = Math.min(peakFlops, this.hbmBandwidth * intensity);

            intensityPoints.push({
                intensity,
                achievableFlops,
                peakFlops,
                memoryBound: intensity < metrics.peakHardwareIntensity
            });
        }

        // Add the current matrix operation as a point
        const matrixPoint = {
            intensity: metrics.arithmeticIntensity,
            achievableFlops: metrics.isComputeBound ?
                peakFlops :
                this.hbmBandwidth * metrics.arithmeticIntensity,
            isCurrentMatrix: true
        };

        return {
            intensityPoints,
            matrixPoint,
            peakHardwareIntensity: metrics.peakHardwareIntensity,
            peakFlops
        };
    }
}

class H100 {
    constructor() {

        this.name = "NVIDIA H100 SXM5";
        this.type = "GPU";

        // Compute capabilities
        this.flopsPerSecondBF16 = 9.89e14;
        this.flopsPerSecondBF16Sparse = 1.979e15;
        this.flopsPerSecondFP32 = 6.7e13;
        this.flopsPerSecondINT8 = 1.979e15;
        this.flopsPerSecondFP8 = 3.958e15;

        // Memory specifications
        this.hbmCapacity = 80e9;
        this.hbmBandwidth = 3.35e12;
        this.l2CacheSize = 50e6;
        this.sharedMemoryPerSM = 228e3;

        // Interconnect specifications
        this.nvlinkBandwidth = 9e10 * 18;
        this.pcieBandwidth = 8e10;

        // Architecture details
        this.tensorCoreConfig = [4, 4, 16];
        this.smCount = 132;
        this.maxGpusPerNode = 256;
        this.nvlinkTopology = "all-to-all";
        this.nvlinkLatency = 0.5e-6;
        this.pcieLatency = 2e-6;

        // Storage for matrices and models
        this.matrices = [];
        this.models = [];
    }

    // Add a matrix to analyze
    addMatrix(type, lhs, rhs, output) {
        const matrix = new Matrix(type, lhs, rhs, output);
        this.matrices.push(matrix);
        return matrix.id;
    }

    // Calculate performance metrics for a matrix
    performanceMetrics(matrixId, multiNode = false, shardingConfig = null) {
        const matrix = this.matrices.find(m => m.id === matrixId);
        if (!matrix) {
            throw new Error(`Matrix with ID ${matrixId} not found`);
        }

        if (multiNode && !shardingConfig) {
            throw new Error("Sharding configuration required for multi-node analysis");
        }

        // Single node performance
        const singleNodeMetrics = this._calculateSingleNodeMetrics(matrix);

        // Multi-node performance if requested
        let multiNodeMetrics = null;
        if (multiNode) {
            multiNodeMetrics = this._calculateMultiNodeMetrics(matrix, shardingConfig);
        }

        return {
            singleNode: singleNodeMetrics,
            multiNode: multiNodeMetrics
        };
    }

    // Calculate single-node performance metrics
    _calculateSingleNodeMetrics(matrix) {
        const B = matrix.lhs.shape[0];
        const D = matrix.lhs.shape[1];
        const F = matrix.rhs.shape[1];

        // Calculate bytes and FLOPs
        const bytes = matrix.calculateBytes();
        const totalFLOPs = matrix.calculateFLOPs();

        // Calculate arithmetic intensity
        const arithmeticIntensity = matrix.calculateArithmeticIntensity();

        // Select appropriate FLOPS rate based on data type
        let flopsPerSecond;
        switch (matrix.lhs.dtype) {
            case "bf16":
                flopsPerSecond = this.flopsPerSecondBF16;
                break;
            case "fp32":
                flopsPerSecond = this.flopsPerSecondFP32;
                break;
            case "int8":
                flopsPerSecond = this.flopsPerSecondINT8;
                break;
            case "fp8":
                flopsPerSecond = this.flopsPerSecondFP8;
                break;
            default:
                flopsPerSecond = this.flopsPerSecondBF16;
        }

        // Calculate peak hardware intensity
        const peakHardwareIntensity = flopsPerSecond / this.hbmBandwidth;

        // Determine if compute or memory bound
        const isComputeBound = arithmeticIntensity > peakHardwareIntensity;

        // Calculate time estimates
        const computeTime = totalFLOPs / flopsPerSecond;
        const memoryTime = bytes.totalBytes / this.hbmBandwidth;

        // Calculate tensor core utilization
        const tcM = this.tensorCoreConfig[0];
        const tcN = this.tensorCoreConfig[1];
        const tcK = this.tensorCoreConfig[2];

        const tensorCoreUtilization = {
            mUtilization: (B % tcM === 0) ? 1.0 : (B / Math.ceil(B / tcM) / tcM),
            nUtilization: (F % tcN === 0) ? 1.0 : (F / Math.ceil(F / tcN) / tcN),
            kUtilization: (D % tcK === 0) ? 1.0 : (D / Math.ceil(D / tcK) / tcK)
        };

        // Calculate SM occupancy (simplified model)
        const warpsPerSM = 64;
        const threadsPerWarp = 32;
        const threadsPerBlock = 256;
        const blocksPerSM = Math.min(16, Math.floor(warpsPerSM * threadsPerWarp / threadsPerBlock));

        // Estimate number of blocks needed for this matmul
        const blocksNeeded = Math.ceil(B / 32) * Math.ceil(F / 32);
        const smOccupancy = Math.min(1.0, blocksNeeded / (blocksPerSM * this.smCount));

        // Calculate L2 cache benefit (simplified model)
        const canFitWeightsInL2 = bytes.rhsBytes <= this.l2CacheSize;
        const l2CacheBenefit = canFitWeightsInL2 ? 1.5 : 1.0;

        // Calculate total time estimates with L2 cache consideration
        const effectiveMemoryTime = memoryTime / l2CacheBenefit;
        const lowerBoundTime = Math.max(computeTime, effectiveMemoryTime);
        const upperBoundTime = computeTime + effectiveMemoryTime;

        return {
            // Basic metrics
            totalFLOPs,
            bytes,
            arithmeticIntensity,
            peakHardwareIntensity,
            isComputeBound,

            // Time estimates
            computeTime,
            memoryTime,
            effectiveMemoryTime,
            lowerBoundTime,
            upperBoundTime,

            // Hardware utilization
            tensorCoreUtilization,
            smOccupancy,
            canFitWeightsInL2,
            l2CacheBenefit
        };
    }

    // Calculate multi-node performance metrics
    _calculateMultiNodeMetrics(matrix, shardingConfig) {
        
        const shardedMatrix = new Matrix(
            matrix.type,
            JSON.parse(JSON.stringify(matrix.lhs)),
            JSON.parse(JSON.stringify(matrix.rhs)),
            JSON.parse(JSON.stringify(matrix.output))
        );

        // Apply sharding to the matrix
        if (shardingConfig.lhsShardDim !== null) {
            shardedMatrix.lhs.shape[shardingConfig.lhsShardDim] =
                Math.ceil(matrix.lhs.shape[shardingConfig.lhsShardDim] / shardingConfig.numDevices);
        }

        if (shardingConfig.rhsShardDim !== null) {
            shardedMatrix.rhs.shape[shardingConfig.rhsShardDim] =
                Math.ceil(matrix.rhs.shape[shardingConfig.rhsShardDim] / shardingConfig.numDevices);
        }

        // Calculate per-device performance
        const perDeviceMetrics = this._calculateSingleNodeMetrics(shardedMatrix);

        // Calculate communication costs
        let communicationCost = 0;

        // For example, if we shard on D dimension, we need to do an all-reduce on the output
        if (shardingConfig.lhsShardDim === 1 || shardingConfig.rhsShardDim === 0) {
            const B = matrix.lhs.shape[0];
            const F = matrix.rhs.shape[1];
            const outputBytes = B * F *
                (matrix.output.dtype === "bf16" ? 2 :
                    matrix.output.dtype === "f32" ? 4 : 1);

            // For H100 with NVSwitch, communication is more efficient
            if (shardingConfig.numDevices <= this.maxGpusPerNode) {

                // Within a single NVSwitch domain - more efficient communication
                const steps = Math.log2(shardingConfig.numDevices);
                communicationCost = (outputBytes / 2) * steps / this.nvlinkBandwidth;

            } else {
                // Across multiple nodes - less efficient
                const nodesNeeded = Math.ceil(shardingConfig.numDevices / this.maxGpusPerNode);
                const intraNodeSteps = Math.log2(this.maxGpusPerNode);
                const interNodeSteps = Math.log2(nodesNeeded);

                const intraNodeCost = (outputBytes / 2) * intraNodeSteps / this.nvlinkBandwidth;
                const interNodeCost = (outputBytes / nodesNeeded) * interNodeSteps / this.pcieBandwidth;

                communicationCost = intraNodeCost + interNodeCost;
            }
        }

        return {
            perDeviceMetrics,
            communicationCost,
            totalTime: Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
            speedupOverSingleDevice: this._calculateSingleNodeMetrics(matrix).lowerBoundTime /
                Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
            shardingEfficiency: perDeviceMetrics.lowerBoundTime /
                Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
            scalingEfficiency: (this._calculateSingleNodeMetrics(matrix).lowerBoundTime /
                Math.max(perDeviceMetrics.lowerBoundTime, communicationCost)) /
                shardingConfig.numDevices
        };
    }

    // Generate recommendations and roofline analysis
    analyze(matrixId) {
        const matrix = this.matrices.find(m => m.id === matrixId);
        if (!matrix) {
            throw new Error(`Matrix with ID ${matrixId} not found`);
        }

        const metrics = this._calculateSingleNodeMetrics(matrix);
        const recommendations = this._generateRecommendations(matrix, metrics);
        const rooflineData = this._generateRooflineData(matrix, metrics);

        return {
            metrics,
            recommendations,
            rooflineData
        };
    }

    // Generate recommendations for improving performance
    _generateRecommendations(matrix, metrics) {
        const recommendations = [];

        // Check if we're memory bound
        if (!metrics.isComputeBound) {
            recommendations.push("Operation is memory-bound. Consider increasing batch size to improve arithmetic intensity.");

            // Recommend FP8 if using bf16
            if (matrix.lhs.dtype === "bf16" && matrix.rhs.dtype === "bf16") {
                recommendations.push("Consider using FP8 precision to reduce memory bandwidth requirements and increase compute throughput.");
            }

            // Recommend sparsity if applicable
            recommendations.push("Consider using structured sparsity to potentially double compute throughput.");
        }

        // Check tensor core utilization
        if (metrics.tensorCoreUtilization.mUtilization < 0.9 ||
            metrics.tensorCoreUtilization.nUtilization < 0.9 ||
            metrics.tensorCoreUtilization.kUtilization < 0.9) {
            recommendations.push(`Consider padding matrix dimensions to multiples of tensor core dimensions (${this.tensorCoreConfig.join('x')}) to improve hardware utilization.`);
        }

        // Check batch size for compute-bound operation
        const B = matrix.lhs.shape[0];
        if (B < 300 && matrix.lhs.dtype === "bf16") {
            recommendations.push("For bf16 matmul on H100, batch size should be at least 300 to be compute-bound.");
        }

        // Check if dimensions are suitable for efficient CUDA kernels
        if (B % 32 !== 0 || matrix.rhs.shape[1] % 32 !== 0 || matrix.lhs.shape[1] % 16 !== 0) {
            recommendations.push("For optimal CUDA kernel performance, consider using dimensions that are multiples of 32 for M and N, and 16 for K.");
        }

        return recommendations;
    }

    // Generate data for roofline plot
    _generateRooflineData(matrix, metrics) {
        // Select appropriate FLOPS rate based on data type
        let flopsPerSecond;
        switch (matrix.lhs.dtype) {
            case "bf16":
                flopsPerSecond = this.flopsPerSecondBF16;
                break;
            case "fp32":
                flopsPerSecond = this.flopsPerSecondFP32;
                break;
            case "int8":
                flopsPerSecond = this.flopsPerSecondINT8;
                break;
            case "fp8":
                flopsPerSecond = this.flopsPerSecondFP8;
                break;
            default:
                flopsPerSecond = this.flopsPerSecondBF16;
        }

        // Generate points for roofline plot
        const intensityPoints = [];

        // Generate logarithmically spaced points for x-axis (arithmetic intensity)
        for (let i = -1; i <= 4; i += 0.1) {
            const intensity = Math.pow(10, i);
            const achievableFlops = Math.min(flopsPerSecond, this.hbmBandwidth * intensity);

            intensityPoints.push({
                intensity,
                achievableFlops,
                peakFlops: flopsPerSecond,
                memoryBound: intensity < metrics.peakHardwareIntensity
            });
        }

        // Add the current matrix operation as a point
        const matrixPoint = {
            intensity: metrics.arithmeticIntensity,
            achievableFlops: metrics.isComputeBound ?
                flopsPerSecond :
                this.hbmBandwidth * metrics.arithmeticIntensity,
            isCurrentMatrix: true
        };

        return {
            intensityPoints,
            matrixPoint,
            peakHardwareIntensity: metrics.peakHardwareIntensity,
            peakFlops: flopsPerSecond
        };
    }
}


const tpu = new TPUv5e();
const gpu = new H100();

const tpuMatrixId = tpu.addMatrix(
  "matmul",
  { shape: [128, 4096], dtype: "bf16" },
  { shape: [4096, 4096], dtype: "bf16" },
  { dtype: "bf16" }
);

const gpuMatrixId = gpu.addMatrix(
  "matmul",
  { shape: [128, 4096], dtype: "bf16" },
  { shape: [4096, 4096], dtype: "bf16" },
  { dtype: "bf16" }
);

const tpuMetrics = tpu.performanceMetrics(tpuMatrixId);
const gpuMetrics = gpu.performanceMetrics(gpuMatrixId);

console.log("TPU v5e Single-Node Performance:", tpuMetrics.singleNode);
console.log("H100 Single-Node Performance:", gpuMetrics.singleNode);

const shardingConfig = {
  numDevices: 4,
  lhsShardDim: 1, 
  rhsShardDim: 0 
};

const tpuMultiNodeMetrics = tpu.performanceMetrics(tpuMatrixId, true, shardingConfig);
const gpuMultiNodeMetrics = gpu.performanceMetrics(gpuMatrixId, true, shardingConfig);

console.log("TPU v5e Multi-Node Performance:", tpuMultiNodeMetrics.multiNode);
console.log("H100 Multi-Node Performance:", gpuMultiNodeMetrics.multiNode);

const tpuAnalysis = tpu.analyze(tpuMatrixId);
const gpuAnalysis = gpu.analyze(gpuMatrixId);

console.log("TPU v5e Recommendations:", tpuAnalysis.recommendations);
console.log("H100 Recommendations:", gpuAnalysis.recommendations);