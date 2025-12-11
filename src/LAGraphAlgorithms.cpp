#include "LAGraphAlgorithms.hpp"
#include <LAGraph.h>

uint64_t lagraph_count_triangles(GrB_Matrix G) {
  if (G == NULL) {
    return 0;
  }

  LAGraph_Graph G_lag = NULL;
  uint64_t ntri = 0;

  if (LAGraph_New(&G_lag, &G, LAGraph_ADJACENCY_UNDIRECTED, NULL) != 0) {
    return 0;
  }

  if (LAGraph_TriangleCount(&ntri, G_lag, NULL) != 0) {
    LAGraph_Delete(&G_lag, NULL);
    return 0;
  }

  LAGraph_Delete(&G_lag, NULL);
  return ntri;
}
