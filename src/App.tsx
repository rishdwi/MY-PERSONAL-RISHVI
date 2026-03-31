/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, LiveServerMessage, Modality, Type } from "@google/genai";
import { AnimatePresence, motion } from "motion/react";
import { Mic, MicOff, Search, Send, Volume2, VolumeX, Youtube, MessageSquare, ExternalLink, Globe } from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";

// --- Constants & Types ---
const MODEL = "gemini-3.1-flash-live-preview";

interface Action {
  type: 'search' | 'open_url' | 'youtube' | 'whatsapp' | 'spotify';
  query?: string;
  url?: string;
  phone?: string;
  message?: string;
}

// --- Helper: Audio Processing ---
function resample(data: Float32Array, fromRate: number, toRate: number): Float32Array {
  const ratio = fromRate / toRate;
  const newLength = Math.ceil(data.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const pos = i * ratio;
    const index = Math.floor(pos);
    const nextIndex = Math.min(index + 1, data.length - 1);
    const fraction = pos - index;
    result[i] = data[index] * (1 - fraction) + data[nextIndex] * fraction;
  }
  return result;
}

function floatTo16BitPCM(float32Array: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

// --- Main Component ---
export default function App() {
  const [isActive, setIsActive] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [response, setResponse] = useState("");
  const [isConnecting, setIsConnecting] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);

  const sessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioQueueRef = useRef<Int16Array[]>([]);
  const isPlayingRef = useRef(false);

  // --- Actions Implementation ---
  const executeAction = useCallback((action: Action) => {
    console.log("Executing action:", action);
    let url = "";
    switch (action.type) {
      case 'search':
        url = `https://www.google.com/search?q=${encodeURIComponent(action.query || "")}`;
        break;
      case 'open_url':
        url = action.url?.startsWith('http') ? action.url : `https://${action.url}`;
        break;
      case 'youtube':
        url = `https://www.youtube.com/results?search_query=${encodeURIComponent(action.query || "")}`;
        break;
      case 'spotify':
        url = `https://open.spotify.com/search/${encodeURIComponent(action.query || "")}`;
        break;
      case 'whatsapp':
        const cleanPhone = action.phone?.replace(/\D/g, '');
        url = `https://wa.me/${cleanPhone}?text=${encodeURIComponent(action.message || "")}`;
        break;
    }

    if (url) {
      window.open(url, '_blank');
    }
  }, []);

  // --- Audio Playback ---
  const playNextInQueue = useCallback(() => {
    if (audioQueueRef.current.length === 0 || isPlayingRef.current) return;

    isPlayingRef.current = true;
    const ctx = audioContextRef.current;
    if (!ctx) return;

    const pcmData = audioQueueRef.current.shift()!;
    const float32Data = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      float32Data[i] = pcmData[i] / 32768.0;
    }

    const buffer = ctx.createBuffer(1, float32Data.length, 24000);
    buffer.getChannelData(0).set(float32Data);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => {
      isPlayingRef.current = false;
      playNextInQueue();
    };
    source.start();
  }, []);

  // --- Session Management ---
  const startSession = async () => {
    if (isActive) return;
    setIsConnecting(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      // Setup Audio Context for recording and playback
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      
      // Resume AudioContext if it's suspended (browser policy)
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const session = await ai.live.connect({
        model: MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: "You are RISHVI, a highly intelligent and friendly browser-based AI assistant. You respond quickly and naturally. You can perform actions like searching the web, opening websites, playing music on YouTube or Spotify, and sending WhatsApp messages. When a user asks for an action, confirm it verbally and then use the provided tools. Keep your verbal responses concise and human-like. If you open a website, say something like 'Opening that for you now.'",
          tools: [
            {
              functionDeclarations: [
                {
                  name: "search_web",
                  description: "Search the web for a query",
                  parameters: {
                    type: Type.OBJECT,
                    properties: { query: { type: Type.STRING } },
                    required: ["query"]
                  }
                },
                {
                  name: "open_website",
                  description: "Open a specific website URL",
                  parameters: {
                    type: Type.OBJECT,
                    properties: { url: { type: Type.STRING } },
                    required: ["url"]
                  }
                },
                {
                  name: "play_music",
                  description: "Play music or search on YouTube or Spotify",
                  parameters: {
                    type: Type.OBJECT,
                    properties: { 
                      query: { type: Type.STRING },
                      platform: { type: Type.STRING, enum: ["youtube", "spotify"] }
                    },
                    required: ["query", "platform"]
                  }
                },
                {
                  name: "send_whatsapp",
                  description: "Send a WhatsApp message to a phone number",
                  parameters: {
                    type: Type.OBJECT,
                    properties: { 
                      phone: { type: Type.STRING, description: "Phone number with country code" },
                      message: { type: Type.STRING }
                    },
                    required: ["phone", "message"]
                  }
                }
              ]
            }
          ],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Kore" } }
          },
          outputAudioTranscription: {},
          inputAudioTranscription: {}
        },
        callbacks: {
          onopen: () => {
            setIsActive(true);
            setIsConnecting(false);
          },
          onmessage: async (msg: LiveServerMessage) => {
            if (msg.serverContent?.modelTurn?.parts) {
              for (const part of msg.serverContent.modelTurn.parts) {
                if (part.inlineData?.data) {
                  const binaryString = atob(part.inlineData.data);
                  const bytes = new Uint8Array(binaryString.length);
                  for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                  }
                  audioQueueRef.current.push(new Int16Array(bytes.buffer));
                  playNextInQueue();
                }
              }
            }

            if (msg.serverContent?.interrupted) {
              audioQueueRef.current = [];
              isPlayingRef.current = false;
            }

            if (msg.toolCall) {
              for (const call of msg.toolCall.functionCalls) {
                const args = call.args as any;
                if (call.name === "search_web") executeAction({ type: 'search', query: args.query });
                if (call.name === "open_website") executeAction({ type: 'open_url', url: args.url });
                if (call.name === "play_music") executeAction({ type: args.platform === 'spotify' ? 'spotify' : 'youtube', query: args.query });
                if (call.name === "send_whatsapp") executeAction({ type: 'whatsapp', phone: args.phone, message: args.message });
                
                // Send tool response back to model
                if (sessionRef.current) {
                  sessionRef.current.sendToolResponse({
                    functionResponses: [{
                      name: call.name,
                      id: call.id,
                      response: { result: "Action executed successfully" }
                    }]
                  });
                }
              }
            }

            // Transcriptions
            if (msg.serverContent?.modelTurn?.parts?.[0]?.text) {
              setResponse(prev => prev + " " + msg.serverContent!.modelTurn!.parts[0].text);
            }
          },
          onclose: () => stopSession(),
          onerror: (err) => {
            console.error("Live API Error:", err);
            stopSession();
          }
        }
      });

      sessionRef.current = session;

      // Send initial greeting to trigger response
      sessionRef.current.sendRealtimeInput({ text: "Hello RISHVI, I'm ready to talk." });

      // Start recording
      const source = audioContextRef.current!.createMediaStreamSource(stream);
      const processor = audioContextRef.current!.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const currentSampleRate = audioContextRef.current!.sampleRate;
        
        // Calculate audio level for UI
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        setAudioLevel(Math.sqrt(sum / inputData.length));

        if (!isMuted && sessionRef.current) {
          // Resample to 16kHz for Gemini
          const resampledData = resample(inputData, currentSampleRate, 16000);
          const pcmBuffer = floatTo16BitPCM(resampledData);
          const base64Data = btoa(String.fromCharCode(...new Uint8Array(pcmBuffer)));
          sessionRef.current.sendRealtimeInput({
            audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
          });
        }
      };

      source.connect(processor);
      processor.connect(audioContextRef.current!.destination);
    } catch (err) {
      console.error("Failed to start session:", err);
      setIsConnecting(false);
    }
  };

  const stopSession = () => {
    sessionRef.current?.close();
    sessionRef.current = null;
    
    processorRef.current?.disconnect();
    processorRef.current = null;
    
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    
    setIsActive(false);
    setAudioLevel(0);
    setTranscript("");
    setResponse("");
  };

  const toggleMute = () => setIsMuted(!isMuted);

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-orange-500/30 flex flex-col items-center justify-center overflow-hidden p-4">
      {/* Background Atmosphere */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-orange-600/10 blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-blue-600/10 blur-[120px] animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      {/* Main Content */}
      <div className="relative z-10 w-full max-w-2xl flex flex-col items-center gap-12">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-2"
        >
          <h1 className="text-6xl font-black tracking-tighter uppercase italic">
            RISHVI
          </h1>
          <p className="text-xs uppercase tracking-[0.3em] text-white/40 font-mono">
            Advanced Browser Intelligence
          </p>
        </motion.div>

        {/* Visualizer / Orb */}
        <div className="relative w-64 h-64 flex items-center justify-center">
          <AnimatePresence>
            {isActive && (
              <motion.div
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0, opacity: 0 }}
                className="absolute inset-0 rounded-full border border-white/10"
              />
            )}
          </AnimatePresence>

          {/* Core Orb */}
          <motion.div
            animate={{
              scale: isActive ? 1 + audioLevel * 1.5 : 1,
              boxShadow: isActive 
                ? `0 0 ${40 + audioLevel * 100}px rgba(242, 125, 38, ${0.2 + audioLevel})`
                : "0 0 20px rgba(255, 255, 255, 0.05)"
            }}
            className={`w-32 h-32 rounded-full flex items-center justify-center transition-colors duration-500 ${
              isActive ? 'bg-orange-500' : 'bg-white/5 border border-white/10'
            }`}
          >
            {isActive ? (
              <Volume2 className="w-8 h-8 text-black animate-pulse" />
            ) : (
              <MicOff className="w-8 h-8 text-white/20" />
            )}
          </motion.div>

          {/* Orbiting Elements (Decorative) */}
          {isActive && (
            <div className="absolute inset-0 animate-spin-slow">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1 h-1 bg-white rounded-full" />
            </div>
          )}
        </div>

        {/* Status & Controls */}
        <div className="flex flex-col items-center gap-8 w-full">
          <div className="flex items-center gap-4">
            <button
              onClick={isActive ? stopSession : startSession}
              disabled={isConnecting}
              className={`px-8 py-4 rounded-full font-bold uppercase tracking-widest text-xs transition-all duration-300 flex items-center gap-3 ${
                isActive 
                  ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-500/20' 
                  : 'bg-white text-black hover:bg-orange-500 hover:text-white'
              } ${isConnecting ? 'opacity-50 cursor-wait' : ''}`}
            >
              {isConnecting ? (
                "Connecting..."
              ) : isActive ? (
                <>
                  <MicOff className="w-4 h-4" /> Stop RISHVI
                </>
              ) : (
                <>
                  <Mic className="w-4 h-4" /> Wake RISHVI
                </>
              )}
            </button>

            {isActive && (
              <button
                onClick={toggleMute}
                className={`p-4 rounded-full border transition-all ${
                  isMuted ? 'bg-orange-500 border-orange-500 text-black' : 'border-white/10 text-white hover:bg-white/5'
                }`}
              >
                {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
              </button>
            )}
          </div>

          {/* Response Text */}
          <div className="w-full min-h-[100px] bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-xl relative overflow-hidden">
            <div className="absolute top-0 left-0 w-1 h-full bg-orange-500" />
            <p className="text-sm text-white/60 font-mono uppercase tracking-wider mb-2">System Response</p>
            <p className="text-lg font-light leading-relaxed">
              {isActive ? (
                response || "Listening for your command..."
              ) : (
                "RISHVI is offline. Click the button above to start your session."
              )}
            </p>
          </div>
        </div>

        {/* Feature Grid (Quick Info) */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full opacity-40 hover:opacity-100 transition-opacity duration-500">
          {[
            { icon: Search, label: "Web Search" },
            { icon: Youtube, label: "YouTube" },
            { icon: MessageSquare, label: "WhatsApp" },
            { icon: Globe, label: "Navigation" }
          ].map((item, i) => (
            <div key={i} className="flex flex-col items-center gap-2 p-4 rounded-xl border border-white/5 bg-white/[0.02]">
              <item.icon className="w-4 h-4" />
              <span className="text-[10px] uppercase tracking-widest font-bold">{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer className="fixed bottom-8 text-[10px] uppercase tracking-[0.4em] text-white/20 font-mono">
        Built for Speed & Intelligence
      </footer>

      <style>{`
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin-slow {
          animation: spin-slow 8s linear infinite;
        }
      `}</style>
    </div>
  );
}
