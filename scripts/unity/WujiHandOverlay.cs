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
// ⚠️ 左右手必须分开对齐,而且不是"两组 Euler"那么简单:
//    Wuji 左右手腕系互为【镜像】(左手 +Y 是手背侧,右手 +Y 是手心侧),
//    旋转矩阵行列式恒为 +1,变不出镜像 —— 所以至少有一只手需要一个轴取反。
//    本脚本默认 Auto Align:从 Captury 手骨的 rest pose 自动解出每只手的旋转,
//    并用【拇指位置】作为判据自动判定该不该镜像。通常不需要手调。
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

    [Header("对齐:自动从 Captury 手骨 rest pose 求解(左右手各解各的)")]
    public bool autoAlign = true;
    [Tooltip("自动把 Wuji 手模型缩放到演员的手长")]
    public bool autoScale = true;

    [Header("手动微调 — 叠加在自动对齐之上,左右手独立")]
    public Vector3 leftEulerOffset = Vector3.zero;
    public Vector3 rightEulerOffset = Vector3.zero;
    [Tooltip("关掉 Auto Align 时才用得上:手动指定该手是否镜像")]
    public bool leftMirrorY = false;
    public bool rightMirrorY = false;
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

    const int THUMB_CMC = 1, INDEX_MCP = 5, MIDDLE_MCP = 9, PINKY_MCP = 17;

    // 每只手一套独立的对齐参数 —— 这是关键,左右手不能共用
    class HandRig
    {
        public string side;
        public Transform hand;
        public List<Transform> joints = new List<Transform>();
        public float[] pos;
        public int[] valid;
        public bool mirror;        // 该手是否需要 Y 轴取反
        public Quaternion align = Quaternion.identity;
        public float scale = 1f;
        public string solveLog = "";
    }

    Doc doc;
    HandRig L, R;
    Animator anim;
    float t0;
    int curFrame;
    readonly Dictionary<string, Transform> map = new Dictionary<string, Transform>();

    void Start()
    {
        doc = JsonUtility.FromJson<Doc>(wujiJson.text);

        foreach (var tr in GetComponentsInChildren<Transform>(true))
            if (!map.ContainsKey(tr.name)) map[tr.name] = tr;

        L = new HandRig { side = "Left", pos = doc.posLeft, valid = doc.validLeft, mirror = leftMirrorY };
        R = new HandRig { side = "Right", pos = doc.posRight, valid = doc.validRight, mirror = rightMirrorY };

        map.TryGetValue(leftHandBone, out L.hand);
        map.TryGetValue(rightHandBone, out R.hand);
        if (L.hand == null || R.hand == null)
        {
            // 前缀猜错时自动兜底:按后缀匹配
            foreach (var kv in map)
            {
                if (L.hand == null && kv.Key.EndsWith(":LeftHand")) L.hand = kv.Value;
                if (R.hand == null && kv.Key.EndsWith(":RightHand")) R.hand = kv.Value;
            }
        }
        Debug.Log($"WujiHandOverlay: LeftHand={(L.hand ? L.hand.name : "未找到")} " +
                  $"RightHand={(R.hand ? R.hand.name : "未找到")}, {doc.frames} 帧 @ {doc.fps} fps");

        foreach (var h in new[] { L, R })
        {
            if (h.hand == null) continue;
            if (autoAlign) Solve(h);
            Build(h, h.side == "Left" ? leftColor : rightColor);
            Debug.Log($"WujiHandOverlay[{h.side}]: mirror={h.mirror} scale={h.scale:F3} " +
                      $"align={h.align.eulerAngles} {h.solveLog}");
        }

        anim = GetComponent<Animator>();
        t0 = Time.time;
    }

    // ---- 自动对齐:左右手各解各的 ----
    // 用掌部(手腕/食指MCP/中指MCP/小指MCP)建正交基求旋转,
    // 再用拇指的落点判定该不该镜像 —— 拇指没参与建基,是独立判据。
    void Solve(HandRig h)
    {
        int f0 = FirstValid(h.valid);
        if (f0 < 0) { h.solveLog = "(无有效帧,退回单位对齐)"; return; }

        // Captury 手骨 rest pose:各掌指关节相对 Hand 骨的位置
        Vector3 cThumb, cIndex, cMiddle, cPinky;
        if (!RestPalm(h, out cThumb, out cIndex, out cMiddle, out cPinky))
        { h.solveLog = "(找不到指骨,退回单位对齐)"; return; }

        Quaternion qChar = PalmBasis(cMiddle, cIndex - cPinky);

        float bestErr = float.MaxValue;
        foreach (bool mir in new[] { false, true })
        {
            Vector3 wWrist = P(h, f0, 0, mir);
            Vector3 wThumb = P(h, f0, THUMB_CMC, mir) - wWrist;
            Vector3 wIndex = P(h, f0, INDEX_MCP, mir) - wWrist;
            Vector3 wMiddle = P(h, f0, MIDDLE_MCP, mir) - wWrist;
            Vector3 wPinky = P(h, f0, PINKY_MCP, mir) - wWrist;

            Quaternion a = qChar * Quaternion.Inverse(PalmBasis(wMiddle, wIndex - wPinky));

            // 残差:四条掌部射线方向的夹角和。拇指是判定镜像的关键项。
            float err = Ang(a * wThumb, cThumb) + Ang(a * wIndex, cIndex)
                      + Ang(a * wMiddle, cMiddle) + Ang(a * wPinky, cPinky);

            if (err < bestErr)
            {
                bestErr = err;
                h.mirror = mir;
                h.align = a;
                h.scale = autoScale && wMiddle.magnitude > 1e-6f
                        ? cMiddle.magnitude / wMiddle.magnitude : 1f;
                h.solveLog = $"残差={err:F1}° (拇指项={Ang(a * wThumb, cThumb):F1}°)";
            }
        }
    }

    // 各掌指关节(Thumb1/Index1/Middle1/Pinky1)在 Hand 骨局部系里的 rest 位置
    bool RestPalm(HandRig h, out Vector3 thumb, out Vector3 index, out Vector3 middle, out Vector3 pinky)
    {
        thumb = index = middle = pinky = Vector3.zero;
        Transform t1, i1, m1, p1;
        string b = h.hand.name;   // 例如 "person:LeftHand"
        if (!map.TryGetValue(b + "Thumb1", out t1) || !map.TryGetValue(b + "Index1", out i1)
         || !map.TryGetValue(b + "Middle1", out m1) || !map.TryGetValue(b + "Pinky1", out p1))
            return false;
        thumb = t1.localPosition; index = i1.localPosition;
        middle = m1.localPosition; pinky = p1.localPosition;
        return true;
    }

    static Quaternion PalmBasis(Vector3 distal, Vector3 radial)
    {
        distal = distal.normalized;
        radial = (radial - Vector3.Dot(radial, distal) * distal).normalized;
        return Quaternion.LookRotation(distal, Vector3.Cross(distal, radial));
    }

    static float Ang(Vector3 a, Vector3 b)
    {
        if (a.sqrMagnitude < 1e-12f || b.sqrMagnitude < 1e-12f) return 0f;
        return Vector3.Angle(a, b);
    }

    static int FirstValid(int[] v)
    {
        if (v == null) return -1;
        for (int i = 0; i < v.Length; i++) if (v[i] == 1) return i;
        return -1;
    }

    // 取第 frame 帧第 joint 个关键点(带镜像)
    static Vector3 P(HandRig h, int frame, int joint, bool mirror)
    {
        int b = (frame * 21 + joint) * 3;
        return new Vector3(h.pos[b], mirror ? -h.pos[b + 1] : h.pos[b + 1], h.pos[b + 2]);
    }

    void Build(HandRig h, Color c)
    {
        var mat = new Material(Shader.Find("Unlit/Color")) { color = c };
        for (int i = 0; i < 21; i++)
        {
            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            s.name = $"wuji_{h.hand.name}_{i}";
            Destroy(s.GetComponent<Collider>());
            s.GetComponent<Renderer>().sharedMaterial = mat;
            s.transform.SetParent(h.hand, false);
            s.transform.localScale = Vector3.one * jointRadius * 2f;
            h.joints.Add(s.transform);
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

        // 左右手各用各的对齐 + 各自的手动微调
        Place(L, Quaternion.Euler(leftEulerOffset));
        Place(R, Quaternion.Euler(rightEulerOffset));
    }

    void Place(HandRig h, Quaternion nudge)
    {
        if (h == null || h.joints.Count == 0 || h.pos == null || h.pos.Length == 0) return;
        bool ok = h.valid != null && curFrame < h.valid.Length && h.valid[curFrame] != 0;
        Quaternion q = nudge * h.align;
        float s = posScale * h.scale;
        for (int i = 0; i < 21; i++)
        {
            int b = (curFrame * 21 + i) * 3;
            if (b + 2 >= h.pos.Length) return;
            h.joints[i].localPosition = q * P(h, curFrame, i, h.mirror) * s;
            h.joints[i].gameObject.SetActive(ok || !hideWhenInvalid);
        }
    }

    void OnDrawGizmos()
    {
        if (!Application.isPlaying || doc == null) return;
        if (L != null) DrawBones(L.joints, leftColor);
        if (R != null) DrawBones(R.joints, rightColor);
#if UNITY_EDITOR
        var anchor = (L != null && L.hand) ? L.hand.position : transform.position;
        bool okL = doc.validLeft != null && curFrame < doc.validLeft.Length && doc.validLeft[curFrame] != 0;
        bool okR = doc.validRight != null && curFrame < doc.validRight.Length && doc.validRight[curFrame] != 0;
        var st = new GUIStyle { fontSize = 14, fontStyle = FontStyle.Bold };
        st.normal.textColor = (okL || okR) ? Color.white : Color.gray;
        UnityEditor.Handles.Label(anchor + Vector3.up * 0.25f,
            $"Wuji  t={curFrame / doc.fps:F2}s  frame {curFrame}  L:{(okL ? "ok" : "--")} R:{(okR ? "ok" : "--")}", st);
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
