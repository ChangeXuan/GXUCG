/*
 *  # 取得顶点的一邻域
 *  把该脚本拖拽到具有网格的GameObject上 
 *  indexMap是全局访问的字典
 *  key是顶点编号，value是该顶点的一邻域编号
 *  test
 *  ```
 *  foreach (int item in indexMap[8]) {
 *      Debug.Log(item);
 *  }
 *  ```
 */
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GetNeighborhood : MonoBehaviour {

    private Mesh mesh;
    // 一邻域字典，key为顶点的num，value为顶点的邻接顶点数组
    public static IDictionary<int, HashSet<int>> indexMap = new Dictionary<int, HashSet<int>>();

    void Start () {
        mesh = GetComponent<MeshFilter>().mesh;
        // mesh.triangles中存的是顶点的索引
        for (int i=0;i<mesh.triangles.Length;i+=3)
        {
            int index = mesh.triangles[i];
            int index1 = mesh.triangles[i+1];
            int index2 = mesh.triangles[i+2];

            saveIndex(index, index1, index2);
            saveIndex(index1, index, index2);
            saveIndex(index2, index, index1);
        }
    }

    void saveIndex(int head, int data1, int data2)
    {
        if (!indexMap.ContainsKey(head))
        {
            indexMap.Add(head, new HashSet<int>());
        }
        indexMap[head].Add(data1);
        indexMap[head].Add(data2);
    }

}
