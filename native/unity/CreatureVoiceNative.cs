// Unity binding for faunavox.dll, from FaunaVox (native/unity). MIT License, Copyright (c) 2026 Russ Martelli.
// Put faunavox.dll in Assets/Plugins/x86_64/ (Windows 64-bit) and this file anywhere under Assets.
//
//   using var voice = NativeCreatureVoice.FromPreset("wolf");
//   voice.Set("mass", creatureMass);                 // genome / body driven
//   voice.SetJson("vowels", "[\"u\",\"o\"]");
//   float[] samples = voice.Render("mating", seed);  // thread-safe; run it off the main thread
//   AudioClip clip = NativeCreatureVoice.ToClip("wolf_mating", samples);  // main thread only
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine;

public static class CreatureVoiceNativeApi
{
    const string Dll = "faunavox";

    [DllImport(Dll)] public static extern IntPtr cv_last_error();
    [DllImport(Dll)] public static extern IntPtr cv_version();
    [DllImport(Dll)] public static extern int cv_sample_rate();
    [DllImport(Dll)] public static extern IntPtr cv_intent_names();
    [DllImport(Dll)] public static extern IntPtr cv_preset_names();

    [DllImport(Dll)] public static extern IntPtr cv_creature_from_json([MarshalAs(UnmanagedType.LPUTF8Str)] string json);
    [DllImport(Dll)] public static extern IntPtr cv_creature_from_preset([MarshalAs(UnmanagedType.LPUTF8Str)] string name);
    [DllImport(Dll)] public static extern IntPtr cv_creature_clone(IntPtr creature);
    [DllImport(Dll)] public static extern void cv_creature_free(IntPtr creature);
    [DllImport(Dll)] public static extern int cv_creature_set(IntPtr creature, [MarshalAs(UnmanagedType.LPUTF8Str)] string field, double value);
    [DllImport(Dll)] public static extern int cv_creature_set_json(IntPtr creature, [MarshalAs(UnmanagedType.LPUTF8Str)] string field, [MarshalAs(UnmanagedType.LPUTF8Str)] string jsonValue);
    [DllImport(Dll)] public static extern IntPtr cv_creature_to_json(IntPtr creature);
    [DllImport(Dll)] public static extern void cv_string_free(IntPtr str);
    [DllImport(Dll)] public static extern IntPtr cv_english_to_phrase([MarshalAs(UnmanagedType.LPUTF8Str)] string text);
    [DllImport(Dll)] public static extern IntPtr cv_creature_from_genome([MarshalAs(UnmanagedType.LPUTF8Str)] string bodyJson, [MarshalAs(UnmanagedType.LPUTF8Str)] string dna, [MarshalAs(UnmanagedType.LPUTF8Str)] string versionsJson);
    [DllImport(Dll)] public static extern IntPtr cv_genome_report(IntPtr creature);
    [DllImport(Dll)] public static extern int cv_field_version();
    [DllImport(Dll)] public static extern int cv_decoder_version();

    [DllImport(Dll)] public static extern IntPtr cv_render(IntPtr creature, [MarshalAs(UnmanagedType.LPUTF8Str)] string intent, ulong seed, float distance, float reverb);
    [DllImport(Dll)] public static extern int cv_buffer_len(IntPtr buffer);
    [DllImport(Dll)] public static extern IntPtr cv_buffer_data(IntPtr buffer);
    [DllImport(Dll)] public static extern int cv_buffer_copy(IntPtr buffer, float[] dst, int capacity);
    [DllImport(Dll)] public static extern void cv_buffer_free(IntPtr buffer);

    public static string LastError() => Marshal.PtrToStringUTF8(cv_last_error());
    public static string Str(IntPtr p) => p == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(p);
}

/// <summary>Owns one native creature description. Render() may be called from several threads
/// at once; Set()/SetJson() must not run while a render of this creature is in flight.</summary>
public sealed class NativeCreatureVoice : IDisposable
{
    public static int SampleRate => CreatureVoiceNativeApi.cv_sample_rate();
    public static string[] Intents => CreatureVoiceNativeApi.Str(CreatureVoiceNativeApi.cv_intent_names()).Split(',');
    public static string[] Presets => CreatureVoiceNativeApi.Str(CreatureVoiceNativeApi.cv_preset_names()).Split(',');

    /// <summary>English text as a phrase for the "speak" intent (CMU dictionary, then letter-to-sound
    /// rules), e.g. "Hey, Heath!" -> "h eh~ih | h iy th". Use it as a call's vowels:
    /// voice.SetJson("calls", "{\"speak\": {\"vowels\": \"" + phrase + "\"}}"). Empty if nothing in the
    /// text can be voiced. The dictionary loads on the first call (about 50 ms).</summary>
    public static string EnglishToPhrase(string text)
    {
        IntPtr s = CreatureVoiceNativeApi.cv_english_to_phrase(text ?? "");
        if (s == IntPtr.Zero) throw new InvalidOperationException(CreatureVoiceNativeApi.LastError());
        try { return CreatureVoiceNativeApi.Str(s); }
        finally { CreatureVoiceNativeApi.cv_string_free(s); }
    }

    IntPtr handle;

    NativeCreatureVoice(IntPtr h)
    {
        if (h == IntPtr.Zero) throw new InvalidOperationException(CreatureVoiceNativeApi.LastError());
        handle = h;
    }

    /// <summary>JSON: {"body": {...}, "voice": {...}}, the same format as create_vocalization.py.</summary>
    public static NativeCreatureVoice FromJson(string json) => new NativeCreatureVoice(CreatureVoiceNativeApi.cv_creature_from_json(json));
    public static NativeCreatureVoice FromPreset(string name) => new NativeCreatureVoice(CreatureVoiceNativeApi.cv_creature_from_preset(name));
    /// <summary>Decode a creature from the voice section of its genome. bodyJson: {"mass": 12.4,
    /// "head_length": 0.14, "neck_length": 0.09, "source": "larynx", "shape": "quadruped", "lips": true,
    /// "teeth": true}. versionsJson null = the versions built into the DLL; store DecoderVersion and
    /// FieldVersion with the creature so a save keeps its voice. "{\"field\": false}" turns steering off.</summary>
    public static NativeCreatureVoice FromGenome(string bodyJson, string dna, string versionsJson = null) =>
        new NativeCreatureVoice(CreatureVoiceNativeApi.cv_creature_from_genome(bodyJson, dna ?? "", versionsJson));
    public static int DecoderVersion => CreatureVoiceNativeApi.cv_decoder_version();
    public static int FieldVersion => CreatureVoiceNativeApi.cv_field_version();

    /// <summary>What the decoder did (JSON), for a creature made by FromGenome: the base voice and each
    /// call on the voice axes, the syllable inventory and phrases as text, badness, versions.</summary>
    public string GenomeReport()
    {
        IntPtr s = CreatureVoiceNativeApi.cv_genome_report(Handle);
        if (s == IntPtr.Zero) throw new InvalidOperationException(CreatureVoiceNativeApi.LastError());
        try { return CreatureVoiceNativeApi.Str(s); }
        finally { CreatureVoiceNativeApi.cv_string_free(s); }
    }

    public NativeCreatureVoice Clone() => new NativeCreatureVoice(CreatureVoiceNativeApi.cv_creature_clone(Handle));

    IntPtr Handle => handle != IntPtr.Zero ? handle : throw new ObjectDisposedException(nameof(NativeCreatureVoice));

    /// <summary>Set a numeric Body/Voice field: mass, tract_length, roughness, pitch, harshness, ...</summary>
    public NativeCreatureVoice Set(string field, double value)
    {
        if (CreatureVoiceNativeApi.cv_creature_set(Handle, field, value) != 0)
            throw new ArgumentException(CreatureVoiceNativeApi.LastError());
        return this;
    }

    /// <summary>Set any field from JSON, e.g. ("vowels", "[\"u\",\"o\"]") or ("source", "\"syrinx\"").</summary>
    public NativeCreatureVoice SetJson(string field, string jsonValue)
    {
        if (CreatureVoiceNativeApi.cv_creature_set_json(Handle, field, jsonValue) != 0)
            throw new ArgumentException(CreatureVoiceNativeApi.LastError());
        return this;
    }

    public string ToJson()
    {
        IntPtr s = CreatureVoiceNativeApi.cv_creature_to_json(Handle);
        try { return CreatureVoiceNativeApi.Str(s); }
        finally { CreatureVoiceNativeApi.cv_string_free(s); }
    }

    /// <summary>Render one vocalization: mono samples at SampleRate, peak-normalized to 0.9.
    /// reverb = 0 gives a dry clip for the game's own mixer (Wwise/FMOD/Unity reverb).</summary>
    public float[] Render(string intent, ulong seed, float distance = 20f, float reverb = 0f)
    {
        IntPtr buf = CreatureVoiceNativeApi.cv_render(Handle, intent, seed, distance, reverb);
        if (buf == IntPtr.Zero) throw new InvalidOperationException(CreatureVoiceNativeApi.LastError());
        try
        {
            var samples = new float[CreatureVoiceNativeApi.cv_buffer_len(buf)];
            CreatureVoiceNativeApi.cv_buffer_copy(buf, samples, samples.Length);
            return samples;
        }
        finally { CreatureVoiceNativeApi.cv_buffer_free(buf); }
    }

    public Task<float[]> RenderAsync(string intent, ulong seed, float distance = 20f, float reverb = 0f) =>
        Task.Run(() => Render(intent, seed, distance, reverb));

    /// <summary>Wrap samples in an AudioClip (main thread only).</summary>
    public static AudioClip ToClip(string name, float[] samples)
    {
        var clip = AudioClip.Create(name, Math.Max(1, samples.Length), 1, SampleRate, false);
        clip.SetData(samples, 0);
        return clip;
    }

    public void Dispose()
    {
        if (handle != IntPtr.Zero) CreatureVoiceNativeApi.cv_creature_free(handle);
        handle = IntPtr.Zero;
        GC.SuppressFinalize(this);
    }

    ~NativeCreatureVoice()
    {
        if (handle != IntPtr.Zero) CreatureVoiceNativeApi.cv_creature_free(handle);
    }
}
