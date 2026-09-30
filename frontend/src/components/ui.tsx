import type { ReactNode } from "react";

export function Section({ title, right, children, className = "" }: { title: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`border-b border-line px-3.5 py-3 ${className}`}>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="panel-title">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Toggle({ label, checked, onChange, hint }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-2 py-[3px] text-[12.5px] text-ink-2 hover:text-ink" title={hint}>
      <span>{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative h-[18px] w-8 shrink-0 rounded-full border transition-colors ${checked ? "border-accent bg-accent/30" : "border-line-2 bg-panel-2"}`}
      >
        <span className={`absolute top-[2px] h-3 w-3 rounded-full transition-all ${checked ? "left-[15px] bg-accent" : "left-[2px] bg-ink-3"}`} />
      </button>
    </label>
  );
}

export function Segmented<T extends string>({ value, options, onChange, size = "sm" }: { value: T; options: { value: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; size?: "sm" | "xs" }) {
  return (
    <div className="flex rounded-md border border-line-2 bg-bg p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={`flex-1 rounded px-1.5 ${size === "xs" ? "py-0.5 text-[10.5px]" : "py-1 text-[11.5px]"} font-medium transition-colors ${value === o.value ? "bg-accent/15 text-accent" : "text-ink-3 hover:text-ink"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Button({ children, onClick, variant = "default", disabled, title, className = "" }: { children: ReactNode; onClick?: () => void; variant?: "default" | "primary" | "danger" | "ghost"; disabled?: boolean; title?: string; className?: string }) {
  const styles = {
    default: "border-line-2 bg-panel-2 text-ink hover:border-ink-3",
    primary: "border-accent bg-accent/15 text-accent hover:bg-accent/25",
    danger: "border-bad/60 bg-bad/10 text-bad hover:bg-bad/20",
    ghost: "border-transparent text-ink-2 hover:text-ink",
  }[variant];
  return (
    <button
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex items-center justify-center gap-1.5 rounded-md border px-2.5 py-1 text-[12px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${styles} ${className}`}
    >
      {children}
    </button>
  );
}

export function Row({ k, v, mono = true, color }: { k: ReactNode; v: ReactNode; mono?: boolean; color?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-[2px] text-[12px]">
      <span className="text-ink-3">{k}</span>
      <span className={`${mono ? "num" : ""} text-right text-ink`} style={color ? { color } : undefined}>
        {v}
      </span>
    </div>
  );
}

export function Dot({ color, size = 9 }: { color: string; size?: number }) {
  return <span className="inline-block shrink-0 rounded-[3px]" style={{ background: color, width: size, height: size }} />;
}
