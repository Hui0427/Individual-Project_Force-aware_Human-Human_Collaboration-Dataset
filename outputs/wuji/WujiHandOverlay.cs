// WujiHandOverlay.cs
// 在 Captury 骨架上叠加显示 Wuji 手套的 21 个关键点。
//
// 用法:
//   1. 这个脚本 + wuji_1448_A.json 一起拖进 Unity Assets
//   2. 脚本挂到该 take 的 person 根物体上(有 Animator 的那个)
//   3. 把 json 拖进 Wuji Json 槽,播放
//
// 手套数据是"手腕局部坐标",手腕的全局位姿由 Captury 提供 —— 所以这个脚本把
// 21 个点作为子物体挂在 Captury 的 LeftHand / RightHand 骨骼下。
//
// ⚠️ 一次性标定:Wuji 的手腕系和 Captury 手骨的朝向不一定一致。播放后如果手指
// 朝向明显不对,调 Wrist Euler Offset(先试 90/180/270 的组合),对了就记下来。
using System;
using System.Collections.Generic;
using UnityEngine;

public class WujiHandOverlay : MonoBehaviour
{
    public TextAsset wujiJson;
    [Header("骨骼名(按你的 take 改前缀)")]
    public string leftHandBone = "person:LeftHand";
    public string rightHandBone = "person:RightHand";

    [Header("显示")]
    public float jointRadius = 0.008f;
    public Color leftColor = new Color(0.2f, 0.6f, 1f);
    public Color rightColor = new Color(1f, 0.35f, 0.25f);
    public bool hideWhenInvalid = true;

    [Header("一次性标定:手指朝向不对就调这里")]
    public Vector3 wristEulerOffset = Vector3.zero;
    public float posScale = 1.0f;

    [Serializable]
    class Doc
    {
        public string shot;
        public float fps;
        public int frames;
        public string[] jointNames;
        public float[] posLeft, posRight;
        public int[] validLeft, validRight;
    }

    // MediaPipe 21 点的连线(用于画骨)
    static readonly int[,] BONES = {
        {0,1},{1,2},{2,3},{3,4},
        {0,5},{5,6},{6,7},{7,8},
        {0,9},{9,10},{10,11},{11,12},
        {0,13},{13,14},{14,15},{15,16},
        {0,17},{17,18},{18,19},{19,20},
        {5,9},{9,13},{13,17}
    };

    Doc doc;
    Transform lHand, rHand;
    readonly List<Transform> lJoints = new List<Transform>();
    readonly List<Transform> rJoints = new List<Transform>();
    Animator anim;
    float t0;
    int curFrame;

    void Start()
    {
        doc = JsonUtility.FromJson<Doc>(wujiJson.text);

        var map = new Dictionary<string, Transform>();
        foreach (var tr in GetComponentsInChildren<Transform>(true))
            if (!map.ContainsKey(tr.name)) map[tr.name] = tr;

        map.TryGetValue(leftHandBone, out lHand);
        map.TryGetValue(rightHandBone, out rHand);
        if (lHand == null || rHand == null)
        {
            // 前缀猜错时自动兜底:按后缀匹配
            foreach (var kv in map)
            {
                if (lHand == null && kv.Key.EndsWith(":LeftHand")) lHand = kv.Value;
                if (rHand == null && kv.Key.EndsWith(":RightHand")) rHand = kv.Value;
            }
        }
        Debug.Log($"WujiHandOverlay: LeftHand={(lHand ? lHand.name : "未找到")} " +
                  $"RightHand={(rHand ? rHand.name : "未找到")}, {doc.frames} 帧 @ {doc.fps} fps");

        if (lHand) Build(lHand, lJoints, leftColor);
        if (rHand) Build(rHand, rJoints, rightColor);

        anim = GetComponent<Animator>();
        t0 = Time.time;
    }

    void Build(Transform parent, List<Transform> into, Color c)
    {
        var mat = new Material(Shader.Find("Unlit/Color")) { color = c };
        for (int i = 0; i < 21; i++)
        {
            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            s.name = $"wuji_{parent.name}_{i}";
            Destroy(s.GetComponent<Collider>());
            s.GetComponent<Renderer>().sharedMaterial = mat;
            s.transform.SetParent(parent, false);
            s.transform.localScale = Vector3.one * jointRadius * 2f;
            into.Add(s.transform);
        }
    }

    void LateUpdate()
    {
        if (doc == null) return;

        float t;
        if (anim && anim.enabled && anim.runtimeAnimatorController != null)
        {
            var st = anim.GetCurrentAnimatorStateInfo(0);
            t = Mathf.Repeat(st.normalizedTime, 1f) * st.length;
        }
        else t = Mathf.Repeat(Time.time - t0, doc.frames / doc.fps);
        curFrame = Mathf.Clamp((int)(t * doc.fps), 0, doc.frames - 1);

        var q = Quaternion.Euler(wristEulerOffset);
        Place(lJoints, doc.posLeft, doc.validLeft, q);
        Place(rJoints, doc.posRight, doc.validRight, q);
    }

    void Place(List<Transform> joints, float[] pos, int[] valid, Quaternion q)
    {
        if (joints.Count == 0 || pos == null || pos.Length == 0) return;
        bool ok = valid != null && curFrame < valid.Length && valid[curFrame] != 0;
        for (int i = 0; i < 21; i++)
        {
            int b = (curFrame * 21 + i) * 3;
            if (b + 2 >= pos.Length) return;
            joints[i].localPosition = q * new Vector3(pos[b], pos[b + 1], pos[b + 2]) * posScale;
            joints[i].gameObject.SetActive(ok || !hideWhenInvalid);
        }
    }

    void OnDrawGizmos()
    {
        if (!Application.isPlaying || doc == null) return;
        DrawBones(lJoints, leftColor);
        DrawBones(rJoints, rightColor);
#if UNITY_EDITOR
        var anchor = lHand ? lHand.position : transform.position;
        bool okL = doc.validLeft != null && curFrame < doc.validLeft.Length && doc.validLeft[curFrame] != 0;
        bool okR = doc.validRight != null && curFrame < doc.validRight.Length && doc.validRight[curFrame] != 0;
        var st = new GUIStyle { fontSize = 14, fontStyle = FontStyle.Bold };
        st.normal.textColor = (okL || okR) ? Color.white : Color.gray;
        UnityEditor.Handles.Label(anchor + Vector3.up * 0.25f,
            $"Wuji  t={curFrame / doc.fps:F2}s  帧{curFrame}  L:{(okL ? "有" : "无")} R:{(okR ? "有" : "无")}", st);
#endif
    }

    void DrawBones(List<Transform> j, Color c)
    {
        if (j.Count < 21) return;
        Gizmos.color = c;
        for (int b = 0; b < BONES.GetLength(0); b++)
        {
            var a = j[BONES[b, 0]]; var d = j[BONES[b, 1]];
            if (a.gameObject.activeSelf && d.gameObject.activeSelf)
                Gizmos.DrawLine(a.position, d.position);
        }
    }
}
