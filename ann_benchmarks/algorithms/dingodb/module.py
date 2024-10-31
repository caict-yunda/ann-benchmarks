import time
import numpy

from multiprocessing.pool import ThreadPool

from dingodb import SDKVectorDingoDB, SDKClient


from ..base.module import BaseANN


ENDPOINT="172.20.3.17:22001"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_PARALLELISM = 1
DEFAULT_BATCH_QUERY_MODE = "separate"

def metric_type_mapping(metric: str):
    metric_type = {"angular": "cosine", "euclidean": "euclidean", "dotproduct": "dotproduct"}.get(metric, None)
    if metric_type is None:
        raise Exception(f"[DingoDB] Not support metric type: {metric}.")
    return metric_type

class DingoHNSW(BaseANN):
    def __init__(self, metric, dim, index_param):
        print(f"index_param: {index_param}")
        self._index_name= f"ann-benchmark-test-{int(time.time())}"
        self._index_config={"efConstruction": index_param["ef_construction"], "maxElements": 600000, "nlinks": index_param["M"]}
        self._metric_type = metric_type_mapping(metric)
        self._dim = dim
        self._batch_size = index_param["batch_size"] if index_param["batch_size"] > 0 else DEFAULT_BATCH_SIZE
        self._parallelism = index_param["parallelism"] if index_param["parallelism"] > 0 else DEFAULT_PARALLELISM
        self._batch_query_mode = DEFAULT_BATCH_QUERY_MODE
        self._sdk_client = SDKClient(ENDPOINT)
        self._sdk_vector_client = SDKVectorDingoDB(self._sdk_client)

    def __str__(self):
        return self.name

    def get_index_param(self):
        return self._index_config

    def set_query_arguments(self, ef):
        self._search_params = {
            "metric_type": self._metric_type,
            "ef": ef
        }

        m = self._index_config.get("nlinks", None)
        ef_construction = self._index_config.get("efConstruction", None)
        self.name = f"DingoDB-HNSW metric: {self._metric_type} M: {m} ef: {ef_construction} search_ef: {ef}"
        
    def create_index(self):
        ret = self._sdk_vector_client.create_index(self._index_name, self._dim, index_type="hnsw",metric_type=self._metric_type,index_config=self._index_config)
        if not ret:
            print(f"[DingoDB] create index {self._index_name} fail.")
            raise Exception("create index fail")

        print(f"[DingoDB] create index {self._index_name} success.")


    def done(self):
        try:
            self._sdk_vector_client.delete_index(self._index_name)
        except RuntimeError as e:
            print(f"[DingoDB] delete vector index({self._index_name}) error: {e}")


    def insert_by_parallel(self, X, parallelism):
        total_size = len(X)

        def add_vector(index, vectors, vector_ids, scalar_data):
            print(f"[DingoDB] Insert data batch size {self._batch_size} schedule {index}/{total_size}")
            self._sdk_vector_client.vector_add(self._index_name, scalar_data, vectors.tolist(), vector_ids)

        pool = ThreadPool(processes=parallelism)
        print(f"[DingoDB] Insert {total_size} data into vector index({self._index_name})")
        for i in range(0, total_size, self._batch_size):
            limit = min(i + self._batch_size, total_size)

            vectors = X[i: limit]
            vector_ids = [i+1 for i in range(i, limit)]
            scalar_data = [{} for i in range(i, limit)]
            
            pool.apply_async(add_vector, (i, vectors, vector_ids, scalar_data,))

        pool.close()
        print(f"[DingoDB] Insert data into vector index({self._index_name}) join....")
        pool.join()

        print(f"[DingoDB] Insert data into vector index({self._index_name}) finish.")

    def insert(self, X):
        if self._parallelism == 1:
            total_size = len(X)
            print(f"[DingoDB] Insert {total_size} data into vector index({self._index_name})")
            for i in range(0, total_size, self._batch_size):
                limit = min(i + self._batch_size, total_size)

                vectors = X[i: limit]
                vector_ids = [i+1 for i in range(i, limit)]
                scalar_data = [{} for i in range(i, limit)]
                
                print(f"[DingoDB] Insert data batch size {self._batch_size} schedule {i}/{total_size}")
                self._sdk_vector_client.vector_add(self._index_name, scalar_data, vectors.tolist(), vector_ids)

            print(f"[DingoDB] Insert data into vector index({self._index_name}) finish.")
        else:
            self.insert_by_parallel(X, self._parallelism)

    def fit(self, X):
        self.create_index()
        
        self.insert(X)

    def query(self, q, k):
        # print(f"[DingoDB] query vector: {q.tolist()} k: {k}")
        result = self._sdk_vector_client.vector_search(self._index_name, q.tolist(), k)
        assert len(result) == 1, f"result size{len(result)} is error."

        # candidate_index is train dataset offset
        candidate_indexs = [ item["id"] - 1 for item in result[0]["vectorWithDistances"]]
        return candidate_indexs

    def batch_query(self, X: numpy.array, n: int) -> None:
        if self._batch_query_mode == "separate":
            # one vector to one request
            pool = ThreadPool()
            self.res = pool.map(lambda q: self.query(q, n), X)
        else:
            # multi vector to one request
            results = self._sdk_vector_client.vector_search(self._index_name, X.tolist(), n)
            assert len(results) == X.size(), f"result size{len(results)} is error."

            multi_candidate_indexs = []
            for result in results:
                one_candidate_indexs = [ item["id"] - 1 for item in result["vectorWithDistances"]]
                multi_candidate_indexs.append(one_candidate_indexs)
            self.res = multi_candidate_indexs

    def get_batch_results(self) -> numpy.array:
        return self.res

