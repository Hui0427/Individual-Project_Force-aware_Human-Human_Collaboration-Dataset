// HandPoseReplayer v3
// 用法:挂在 person1 的 FBX 实例根物体上,把 replayer_data.json 拖进 Pose Json 槽,播放。
//
// 这一版 Animator 保持开启:身体、走位、Hips 全部还是你原来的动画在驱动,
// 脚本只在每帧渲染前覆盖"Hand"相关的 32 根骨骼的旋转 —— 人的位置不可能再变。
// 帧号自动跟 Animator 当前播放进度同步。
//
// Game 视图左上角会显示当前秒数/帧号,进入优化段(默认 150~270 帧,即 2.5~4.5s)时高亮提示。
using System;
using System.Collections.Generic;
using UnityEngine;

public class HandPoseReplayer : MonoBehaviour
{
    public TextAsset poseJson;
    [Header("勾=优化后的手,不勾=原始(播放中随时切)")]
    public bool playRefined = true;
    public bool mirrorRotations = true;
    [Header("优化段帧范围(见 skeleton_meta.json 的 frames)")]
    public int refinedStart = 150;
    public int refinedEnd = 270;

    [Header("定格对比:勾上后画面停在 Freeze Frame,再切 Play Refined 看差别")]
    public bool freeze = false;
    [Range(0, 1108)] public int freezeFrame = 262;   // 帧262 = 食指尖差别最大(6cm)
    int lastApplied = -1;

    [Header("诊断:勾上后所有匹配到的手骨大幅摆动。手不动 = 名字没匹配上")]
    public bool testWiggle = false;

    [Serializable]
    class PoseData
    {
        public float fps;
        public int frames;
        public string[] names;
        public string hipsName;
        public float[] qOrig;
        public float[] q;
        public float[] hipsT;
    }

    PoseData data;
    readonly Dictionary<string, Transform> bones = new Dictionary<string, Transform>();
    readonly List<int> handIdx = new List<int>();
    Animator anim;
    float fallbackT0;
    int curFrame;
    GUIStyle style;

    void Start()
    {
        data = JsonUtility.FromJson<PoseData>(poseJson.text);
        foreach (var tr in GetComponentsInChildren<Transform>(true))
            if (!bones.ContainsKey(tr.name)) bones[tr.name] = tr;

        for (int n = 0; n < data.names.Length; n++)
            if (data.names[n].Contains("Hand"))
                handIdx.Add(n);

        anim = GetComponent<Animator>();   // 保持开启,身体和位置归它管
        fallbackT0 = Time.time;

        // 关键诊断:数据里的骨骼名和场景里的 Transform 名到底对不对得上
        int matched = 0;
        var missing = new List<string>();
        foreach (int n in handIdx)
            if (bones.ContainsKey(data.names[n])) matched++;
            else if (missing.Count < 5) missing.Add(data.names[n]);
        Debug.Log($"HandPoseReplayer: 手部骨骼匹配 {matched}/{handIdx.Count}");
        if (matched < handIdx.Count)
        {
            var actual = new List<string>();
            foreach (var kv in bones)
                if (kv.Key.Contains("Hand") && actual.Count < 5) actual.Add(kv.Key);
            Debug.LogWarning("HandPoseReplayer: 数据里找不到的名字如 [" + string.Join(", ", missing) +
                             "];场景里实际的手骨名如 [" + string.Join(", ", actual) + "]");
        }
    }

    void LateUpdate()
    {
        if (data == null) return;

        bool hasAnim = anim != null && anim.enabled && anim.runtimeAnimatorController != null;

        if (freeze)
        {
            // 冻结:Animator 跳到指定帧并停住,手也用同一帧 -> 全身定格,可拖 slider 逐帧看
            curFrame = Mathf.Clamp(freezeFrame, 0, data.frames - 1);
            if (hasAnim)
            {
                anim.speed = 0f;
                if (lastApplied != curFrame)
                {
                    var st0 = anim.GetCurrentAnimatorStateInfo(0);
                    anim.Play(st0.fullPathHash, 0, (curFrame / data.fps) / st0.length);
                    lastApplied = curFrame;
                }
            }
        }
        else
        {
            if (hasAnim && anim.speed == 0f) anim.speed = 1f;
            lastApplied = -1;
            float t;
            if (hasAnim)
            {
                var st = anim.GetCurrentAnimatorStateInfo(0);
                t = Mathf.Repeat(st.normalizedTime, 1f) * st.length;   // 跟随 Animator 进度
            }
            else
            {
                t = Mathf.Repeat(Time.time - fallbackT0, data.frames / data.fps);
            }
            curFrame = Mathf.Clamp((int)(t * data.fps), 0, data.frames - 1);
        }

        var Q = playRefined ? data.q : data.qOrig;
        int nb = data.names.Length;
        foreach (int n in handIdx)
        {
            Transform tr;
            if (!bones.TryGetValue(data.names[n], out tr)) continue;
            int i = (curFrame * nb + n) * 4;
            float x = Q[i], y = Q[i + 1], z = Q[i + 2], w = Q[i + 3];
            var q = mirrorRotations
                ? new Quaternion(x, -y, -z, w)
                : new Quaternion(x, y, z, w);
            if (testWiggle)
                q = q * Quaternion.Euler(Mathf.Sin(Time.time * 6f) * 40f, 0, 0);
            tr.localRotation = q;
        }
    }

    string Status()
    {
        bool inSeg = curFrame >= refinedStart && curFrame < refinedEnd;
        string mode = playRefined ? "优化后" : "原始";
        return $"t = {curFrame / data.fps:F2} s   帧 {curFrame}   [{mode}]"
               + (inSeg ? "   ◀◀ 优化段!盯手指看" : "");
    }

    void OnGUI()   // Game 视图左上角
    {
        if (data == null) return;
        if (style == null)
            style = new GUIStyle(GUI.skin.label) { fontSize = 22, fontStyle = FontStyle.Bold };
        style.normal.textColor =
            (curFrame >= refinedStart && curFrame < refinedEnd) ? Color.yellow : Color.white;
        GUI.Label(new Rect(12, 8, 900, 40), Status(), style);
    }

#if UNITY_EDITOR
    void OnDrawGizmos()   // Scene 视图:人头顶悬浮同一行字
    {
        if (!Application.isPlaying || data == null) return;
        Transform hips;
        var anchor = bones.TryGetValue(data.hipsName, out hips)
            ? hips.position : transform.position;
        var s = new GUIStyle { fontSize = 16, fontStyle = FontStyle.Bold };
        s.normal.textColor =
            (curFrame >= refinedStart && curFrame < refinedEnd) ? Color.yellow : Color.white;
        UnityEditor.Handles.Label(anchor + Vector3.up * 1.2f, Status(), s);
    }
#endif
}
