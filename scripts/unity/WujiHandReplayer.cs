// WujiHandReplayer
// 把 Wuji 手套的手指动作接到 Captury 骨架的手腕上。
//
// 用法:
//   1. 用 wuji_to_unity.py 导出 JSON,拖进 Wuji Json 槽
//   2. 把本脚本挂在 Captury 的 person FBX 实例根物体上
//   3. Bone Prefix 填 FBX 里的命名空间(如 "person:" / "person1:")
//   4. 播放。手指反向弯曲 -> 勾 Mirror Y
//
// Animator 保持开启:身体、走位、手腕全部还是 Captury 的动画在驱动,
// 本脚本只在 LateUpdate 里覆盖每只手 15 根指骨的旋转。
using System;
using System.Collections.Generic;
using UnityEngine;

public class WujiHandReplayer : MonoBehaviour
{
    [Header("wuji_to_unity.py 导出的 JSON")]
    public TextAsset wujiJson;

    [Header("Captury FBX 的骨骼命名空间")]
    public string bonePrefix = "person:";

    public bool driveLeft = true;
    public bool driveRight = true;

    [Header("手指朝手背方向反弯 -> 勾对应那只手")]
    [Tooltip("左右手的 rest pose 互为镜像,通常需要且只需要给其中一只勾上")]
    public bool mirrorLeft = false;
    public bool mirrorRight = false;

    [Range(0f, 1f)] public float weight = 1f;
    [Tooltip("手动微调帧对齐(单位:帧,正=手指数据往后挪)")]
    public int frameOffset = 0;

    [Header("调试")]
    public bool showHud = true;
    public bool freeze = false;
    public int freezeFrame = 0;

    // ---- JSON 结构(与 wuji_to_unity.py 对应) ----
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

    // Wuji 21 点里,每根手指要驱动的 3 个关节 + 末端(只作瞄准目标)
    static readonly int[][] FingerJointIdx =
    {
        new[] { 1, 2, 3, 4 },      // thumb : cmc, mcp, ip, tip
        new[] { 5, 6, 7, 8 },      // index : mcp, pip, dip, tip
        new[] { 9, 10, 11, 12 },   // middle
        new[] { 13, 14, 15, 16 },  // ring
        new[] { 17, 18, 19, 20 },  // pinky
    };
    static readonly string[] FingerName = { "Thumb", "Index", "Middle", "Ring", "Pinky" };

    const int MIDDLE_MCP = 9, INDEX_MCP = 5, PINKY_MCP = 17;

    class Hand
    {
        public string side;
        public Transform wrist;
        public Transform[,] bone = new Transform[5, 4];  // [手指, 关节] 第4个是 EE
        public Quaternion align;                          // wuji 手腕系 -> 手骨局部系
        public float[] pos;
        public int[] valid;
        public bool mirror;
        public bool ok;
    }

    Doc doc;
    readonly List<Hand> hands = new List<Hand>();
    readonly Dictionary<string, Transform> bones = new Dictionary<string, Transform>();
    Animator anim;
    int curFrame;
    GUIStyle style;

    void Awake()
    {
        if (wujiJson == null) { Debug.LogError("WujiHandReplayer: 没有指定 JSON"); return; }
        doc = JsonUtility.FromJson<Doc>(wujiJson.text);
        anim = GetComponent<Animator>();

        foreach (var tr in GetComponentsInChildren<Transform>(true))
            if (!bones.ContainsKey(tr.name)) bones[tr.name] = tr;

        if (driveLeft) hands.Add(Build("Left", doc.posLeft, doc.validLeft, mirrorLeft));
        if (driveRight) hands.Add(Build("Right", doc.posRight, doc.validRight, mirrorRight));
    }

    Hand Build(string side, float[] pos, int[] valid, bool mirror)
    {
        var h = new Hand { side = side, pos = pos, valid = valid, mirror = mirror };
        if (pos == null || pos.Length == 0)
        {
            Debug.LogWarning($"WujiHandReplayer: JSON 里没有 {side} 手数据");
            return h;
        }

        h.wrist = Find(bonePrefix + side + "Hand");
        if (h.wrist == null) { Report(side); return h; }

        for (int f = 0; f < 5; f++)
            for (int j = 0; j < 4; j++)
            {
                string n = bonePrefix + side + "Hand" + FingerName[f] + (j < 3 ? (j + 1).ToString() : "EE");
                h.bone[f, j] = Find(n);
                if (h.bone[f, j] == null) { Debug.LogWarning($"WujiHandReplayer: 找不到骨骼 {n}"); Report(side); return h; }
            }

        // "wuji 手腕系 -> 手骨局部系" 的常量旋转。
        // 两侧都用同一套方式从掌部构造正交基:掌骨方向(wrist->middle_mcp) + 掌宽方向(pinky_mcp->index_mcp)。
        // 掌部在 wuji 模型里是刚体(实测逐帧 std = 0),所以随便取一帧都行。
        var rest = RestPoseInWristSpace(h);
        Quaternion qChar = PalmBasis(rest[MIDDLE_MCP], rest[INDEX_MCP] - rest[PINKY_MCP]);

        int f0 = FirstValidFrame(valid);
        if (f0 < 0) { Debug.LogWarning($"WujiHandReplayer: {side} 手没有任何有效帧"); return h; }
        int b0 = f0 * 21 * 3;
        Vector3 wMid = W(h, b0, MIDDLE_MCP) - W(h, b0, 0);
        Quaternion qWuji = PalmBasis(wMid, W(h, b0, INDEX_MCP) - W(h, b0, PINKY_MCP));

        h.align = qChar * Quaternion.Inverse(qWuji);
        h.ok = true;
        return h;
    }

    // 由掌骨方向(前)和掌宽方向(右)构造一个正交基
    static Quaternion PalmBasis(Vector3 distal, Vector3 radial)
    {
        distal = distal.normalized;
        radial = (radial - Vector3.Dot(radial, distal) * distal).normalized;
        return Quaternion.LookRotation(distal, Vector3.Cross(distal, radial));
    }

    static int FirstValidFrame(int[] valid)
    {
        for (int i = 0; i < valid.Length; i++) if (valid[i] == 1) return i;
        return -1;
    }

    // 从 bind pose 的 localPosition/localRotation 递推出各关节在手腕局部系里的静止位置
    Dictionary<int, Vector3> RestPoseInWristSpace(Hand h)
    {
        var rest = new Dictionary<int, Vector3>();
        for (int f = 0; f < 5; f++)
        {
            Vector3 p = Vector3.zero;
            Quaternion q = Quaternion.identity;
            for (int j = 0; j < 4; j++)
            {
                var tr = h.bone[f, j];
                p += q * tr.localPosition;
                q *= tr.localRotation;
                rest[FingerJointIdx[f][j]] = p;
            }
        }
        return rest;
    }

    Transform Find(string n) { Transform t; return bones.TryGetValue(n, out t) ? t : null; }

    void Report(string side)
    {
        var sample = new List<string>();
        foreach (var kv in bones) if (kv.Key.Contains("Hand") && sample.Count < 6) sample.Add(kv.Key);
        Debug.LogWarning($"WujiHandReplayer: {side} 手骨骼没匹配上。Bone Prefix = \"{bonePrefix}\";" +
                         $"场景里实际的手骨名例如 [{string.Join(", ", sample)}]");
    }

    void LateUpdate()   // 必须在 Animator / Captury 写完骨架之后
    {
        if (doc == null) return;

        if (freeze)
        {
            curFrame = Mathf.Clamp(freezeFrame, 0, doc.frames - 1);
            if (anim != null && anim.enabled) anim.speed = 0f;
        }
        else
        {
            if (anim != null && anim.enabled && anim.speed == 0f) anim.speed = 1f;
            float t = Time.time;
            if (anim != null && anim.enabled && anim.runtimeAnimatorController != null)
            {
                var st = anim.GetCurrentAnimatorStateInfo(0);
                t = Mathf.Repeat(st.normalizedTime, 1f) * st.length;
            }
            curFrame = Mathf.Clamp(Mathf.RoundToInt(t * doc.fps) + frameOffset, 0, doc.frames - 1);
        }

        foreach (var h in hands)
        {
            if (!h.ok || h.valid == null || curFrame >= h.valid.Length || h.valid[curFrame] == 0) continue;
            Apply(h, curFrame);
        }
    }

    void Apply(Hand h, int frame)
    {
        int baseIdx = frame * 21 * 3;
        Quaternion wristRot = h.wrist.rotation;

        for (int f = 0; f < 5; f++)
        {
            for (int j = 0; j < 3; j++)   // 只驱动 1/2/3,EE 仅作瞄准目标
            {
                Transform bone = h.bone[f, j], child = h.bone[f, j + 1];

                Vector3 want = W(h, baseIdx, FingerJointIdx[f][j + 1]) - W(h, baseIdx, FingerJointIdx[f][j]);
                if (want.sqrMagnitude < 1e-10f) continue;

                Vector3 target = wristRot * (h.align * want.normalized);
                Vector3 current = child.position - bone.position;
                if (current.sqrMagnitude < 1e-10f) continue;

                Quaternion delta = Quaternion.FromToRotation(current.normalized, target);
                Quaternion goal = delta * bone.rotation;
                bone.rotation = weight >= 1f ? goal : Quaternion.Slerp(bone.rotation, goal, weight);
            }
        }
    }

    // 取第 joint 个关键点(已按该手的 mirror 设置处理)
    static Vector3 W(Hand h, int baseIdx, int joint)
    {
        int i = baseIdx + joint * 3;
        return new Vector3(h.pos[i], h.mirror ? -h.pos[i + 1] : h.pos[i + 1], h.pos[i + 2]);
    }

    void OnGUI()
    {
        if (!showHud || doc == null) return;
        if (style == null) style = new GUIStyle(GUI.skin.label) { fontSize = 20, fontStyle = FontStyle.Bold };
        var s = $"{doc.shot}  帧 {curFrame}/{doc.frames}  t={curFrame / doc.fps:F2}s";
        foreach (var h in hands)
        {
            bool v = h.ok && h.valid != null && curFrame < h.valid.Length && h.valid[curFrame] == 1;
            s += $"   {h.side}:{(h.ok ? (v ? "手套" : "无数据") : "未绑定")}";
        }
        style.normal.textColor = Color.white;
        GUI.Label(new Rect(12, 8, 1200, 40), s, style);
    }
}
